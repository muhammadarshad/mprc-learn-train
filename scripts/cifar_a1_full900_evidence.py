"""CIFAR A1: full-source local evidence accumulation under 1,808-byte pass budget.

Each observation pass contains at most:
    113 spatial centers x 16 byte channels = 1,808 bytes.

The complete valid 3x3-center field is 30x30 = 900 source centers.
We traverse it in generator-7 order and split it into 8 passes:
    7 full passes of 113 + final pass of 109 = 900 centers.

Critically, passes do NOT overwrite one another.  Each pass contributes integer
class evidence to a wide decision accumulator. This separates:
  transient observation state (<=1,808 bytes)
from
  accumulated class decision evidence (10 wide integers).

Local evidence is a shared per-channel/per-bit 9-bit pattern model, so the
classifier learns which local shapes are class-relevant without requiring
900 position-specific tables. Existing regional/global v0 branches provide
coarse position/extent evidence.

Promotion target: shadow accuracy > 50%.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request,time,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10";CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz";MD5="c58f30108f718f92721af3b95e74349a"
K=10;SEED=20260927;ALPHA=1
TARGET=0.50
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]
NEIGH=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
GEN=7;DOMAIN=900
assert math.gcd(GEN,DOMAIN)==1
ORDER=np.asarray([(GEN*t)%DOMAIN for t in range(DOMAIN)],dtype=np.int16)
ROWS=(ORDER//30+1).astype(np.int16);COLS=(ORDER%30+1).astype(np.int16)
PASSES=[slice(i,min(i+113,900)) for i in range(0,900,113)]
assert [s.stop-s.start for s in PASSES]==[113,113,113,113,113,113,113,109]

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for z in iter(lambda:f.read(1<<20),b""):h.update(z)
 return h.hexdigest()
def data_dir():
 if not ARCH.exists() or md5(ARCH)!=MD5:urllib.request.urlretrieve(URL,ARCH)
 d=CACHE/"cifar-10-batches-py"
 if not d.exists():
  with tarfile.open(ARCH,"r:gz") as tf:tf.extractall(CACHE)
 return d
def batch(p):
 with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
 return (np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy(),
         np.asarray(d[b"labels"],dtype=np.int64))
def load():
 d=data_dir();xs=[];ys=[]
 for i in range(1,6):
  x,y=batch(d/f"data_batch_{i}");xs.append(x);ys.append(y)
 tx=np.concatenate(xs);ty=np.concatenate(ys);vx,vy=batch(d/"test_batch")
 return tx,ty,vx,vy
def split(y):
 rng=np.random.default_rng(SEED);fit=[];val=[];shadow=[]
 for c in range(10):
  ix=np.flatnonzero(y==c);ix=ix[rng.permutation(len(ix))]
  val.extend(ix[:500]);shadow.extend(ix[500:1000]);fit.extend(ix[1000:])
 return map(lambda z:np.asarray(sorted(z),dtype=np.int64),(fit,val,shadow))
def luma(x):
 r=x[...,0].astype(np.uint16);g=x[...,1].astype(np.uint16);b=x[...,2].astype(np.uint16)
 return ((77*r+150*g+29*b)>>8).astype(np.uint8)
def blur(a,r):
 p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
 I=np.pad(p,((0,0),(1,0),(1,0)),mode="constant").cumsum(1,dtype=np.uint32).cumsum(2,dtype=np.uint32)
 k=2*r+1;s=I[:,k:,k:]-I[:,:-k,k:]-I[:,k:,:-k]+I[:,:-k,:-k]
 return ((s+k*k//2)//(k*k)).astype(np.uint8)
def fields(x):
 R=x[...,0];G=x[...,1];B=x[...,2];Y=luma(x)
 Gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
 Chroma=(x.max(-1).astype(np.int16)-x.min(-1).astype(np.int16)).astype(np.uint8)
 gx=np.zeros_like(Y,dtype=np.int16);gy=np.zeros_like(Y,dtype=np.int16)
 gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
 gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)
 Gx=np.clip(128+gx//2,0,255).astype(np.uint8);Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
 Grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)
 lap=np.zeros_like(Y,dtype=np.int16);c=Y[:,1:-1,1:-1].astype(np.int16)
 lap[:,1:-1,1:-1]=Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c
 Lap=np.clip(128+lap//4,0,255).astype(np.uint8)
 b1=blur(Y,1);b2=blur(Y,2);b4=blur(Y,4);b8=blur(Y,8)
 H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8);H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
 M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8);L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
 Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 dgy=np.zeros_like(Y,dtype=np.int16);dgx=np.zeros_like(Y,dtype=np.int16)
 dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2];dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
 Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
 return dict(zip(CHANNEL_NAMES,[R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl]))

def patterns(a,bit):
 z=((a>>bit)&1).astype(np.uint8)
 s=z[:,ROWS,COLS].astype(np.uint16)
 for k,(dr,dc) in enumerate(NEIGH,1):s|=z[:,ROWS+dr,COLS+dc].astype(np.uint16)<<k
 return s

# shared categorical local model: [16,8,512,10] class evidence
def fit_local(F,y_all,fit_idx):
 tables=np.zeros((16,8,512,10),dtype=np.int32)
 for ci,name in enumerate(CHANNEL_NAMES):
  a=F[name]
  for bit in range(8):
   p=patterns(a,bit)
   cnt=np.zeros((512,10),dtype=np.int64)
   for c in range(10):
    vals=p[y_all[fit_idx]==c].reshape(-1)
    cnt[:,c]=np.bincount(vals,minlength=512)
   total=cnt.sum(1,keepdims=True)
   # integer centered evidence; no log/float model state
   tables[ci,bit]=(10*cnt-total).astype(np.int32)
 return tables

def score_local(F,tables,ids):
 S=np.zeros((len(ids),10),dtype=np.int64)
 for ci,name in enumerate(CHANNEL_NAMES):
  a=F[name][ids]
  for bit in range(8):
   p=patterns(a,bit)
   tab=tables[ci,bit]
   # process in 113-anchor capacity passes, accumulate decision evidence
   for sl in PASSES:
    vals=p[:,sl]
    S+=tab[vals].sum(axis=1,dtype=np.int64)
 return S

# lightweight global raw-byte histograms to preserve coarse colour/field distribution.
def fit_global(F,y_all,fit_idx):
 out={}
 for ci,name in enumerate(CHANNEL_NAMES):
  a=F[name]
  # 16 ring bins from high nibble; class evidence.
  cnt=np.zeros((16,10),dtype=np.int64)
  for c in range(10):
   vals=(a[y_all[fit_idx]==c]>>4).reshape(-1)
   cnt[:,c]=np.bincount(vals,minlength=16)
  out[name]=(10*cnt-cnt.sum(1,keepdims=True)).astype(np.int32)
 return out
def score_global(F,tables,ids):
 S=np.zeros((len(ids),10),dtype=np.int64)
 for name in CHANNEL_NAMES:
  vals=(F[name][ids]>>4).reshape(len(ids),-1)
  S+=tables[name][vals].sum(axis=1,dtype=np.int64)
 return S
def acc(S,y):return float(np.mean(S.argmax(1)==y))

t0=time.time()
tx,ty,vx,vy=load();fit,val,shadow=tuple(split(ty));TF=fields(tx);VF=fields(vx)
lt=fit_local(TF,ty,fit);gt=fit_global(TF,ty,fit)
VL=score_local(TF,lt,val);SL=score_local(TF,lt,shadow);TL=score_local(VF,lt,np.arange(len(vx)))
VG=score_global(TF,gt,val);SG=score_global(TF,gt,shadow);TG=score_global(VF,gt,np.arange(len(vx)))

best=None
for wl in (1,2,4,8):
 for wg in (0,1,2,4,8):
  A=wl*VL+wg*VG;a=acc(A,ty[val]);key=(a,-wl-wg,-wl,-wg)
  if best is None or key>best[0]:best=(key,(wl,wg),a)
(wl,wg),va=best
sha=acc(wl*SL+wg*SG,ty[shadow]);tea=acc(wl*TL+wg*TG,vy)
status="PASS_A1" if sha>TARGET else "FAIL_A1"
R={"model":"MPRC-CIFAR-A1-full900-evidence","status":status,
   "target":{"shadow":TARGET},"passes":[s.stop-s.start for s in PASSES],
   "observation_pass_bytes":1808,"valid_kernel_centers":900,
   "learning":{"local":"shared integer class evidence over joint 9-bit states","global":"16-bin byte occupancy evidence","softmax":False,"gradient_descent":False},
   "validation":{"accuracy":va,"weights":[wl,wg]},
   "shadow":{"accuracy":sha,"used_for_selection":False},
   "test":{"accuracy":tea,"status":"exploratory; test history exposed"},
   "runtime_seconds":time.time()-t0,
   "claim_boundary":"A1 tests full-source evidence accumulation while respecting the 1,808-byte transient observation budget. Decision accumulator is not image storage."}
out=ROOT/"results"/"cifar_a1_full900_evidence.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps(R,indent=2),flush=True)
