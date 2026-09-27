"""CIFAR A0: capacity-aligned 113-anchor x 16-channel MPRC evidence classifier.

Goal
----
First accuracy rung: beat the historical 46.68% CIFAR test baseline.

Corrected observation budget
----------------------------
113 spatial anchors x 16 byte channels = 1,808 observed Z256 bytes.
Those 1,808 bytes correspond to the corrected 14,464-bit storage capacity.

The eight bit planes are DERIVED from each stored byte; they are not extra
storage states.

Spatial anchors
---------------
Use a generator-7 walk over the 30x30 valid 3x3-center domain:
    p_t = 7 t mod 900, t=0..112.
Because gcd(7,900)=1, all 113 selected centers are unique.

LOCAL
-----
For each channel byte and bit plane, encode one joint 9-bit state:
    center + 8 ordered neighbours -> state 0..511.
This preserves the full 1+8 local pattern at each of 113 anchors.

REGIONAL/GLOBAL
---------------
Reuse the established v0 native 7x16 / 16x7 regional evidence and whole-image
global evidence. These are evidence computations, not claims of additional
stored image pixels.

LEARN
-----
Categorical class-evidence LUTs populated from the 40k fit split.
Validation selects only integer branch loudnesses. A 5k shadow split is never
used for selection. No softmax or gradient descent.

Promotion
---------
A0 passes only if shadow accuracy > 46.68%. Official test is exploratory because
this repository's CIFAR test history is already exposed.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request,time,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10";CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz";MD5="c58f30108f718f92721af3b95e74349a"

K=10
ALPHA=0.05
LOG_SCALE=16
SEED=20260927
TARGET=0.4668
CHANNEL_NAMES=[
 "R","G","B","Gray","Luma","Chroma","Gx","Gy",
 "Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"
]

assert len(CHANNEL_NAMES)==16
assert 16*113==1808
assert 1808*8==14464

# 30x30 valid center domain for a 3x3 kernel.
GEN=7
DOMAIN=30*30
assert math.gcd(GEN,DOMAIN)==1
flat=np.asarray([(GEN*t)%DOMAIN for t in range(113)],dtype=np.int16)
CENTER_ROWS=(flat//30+1).astype(np.int16)
CENTER_COLS=(flat%30+1).astype(np.int16)
assert len(set(zip(map(int,CENTER_ROWS),map(int,CENTER_COLS))))==113

NEIGHBORS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def md5(path):
 h=hashlib.md5()
 with open(path,"rb") as f:
  for z in iter(lambda:f.read(1<<20),b""):h.update(z)
 return h.hexdigest()

def ensure_dataset():
 if not ARCHIVE.exists() or md5(ARCHIVE)!=MD5:urllib.request.urlretrieve(URL,ARCHIVE)
 assert md5(ARCHIVE)==MD5
 d=CACHE/"cifar-10-batches-py"
 if not d.exists():
  with tarfile.open(ARCHIVE,"r:gz") as tf:tf.extractall(CACHE)
 return d

def load_batch(path):
 with open(path,"rb") as f:d=pickle.load(f,encoding="bytes")
 x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
 y=np.asarray(d[b"labels"],dtype=np.int64)
 return x,y

def load_cifar():
 d=ensure_dataset();xs=[];ys=[]
 for i in range(1,6):
  x,y=load_batch(d/f"data_batch_{i}");xs.append(x);ys.append(y)
 tx=np.concatenate(xs);ty=np.concatenate(ys)
 vx,vy=load_batch(d/"test_batch")
 return tx,ty,vx,vy

def split_40_5_5(y):
 rng=np.random.default_rng(SEED);fit=[];val=[];shadow=[]
 for c in range(K):
  ix=np.flatnonzero(y==c);ix=ix[rng.permutation(len(ix))]
  val.extend(ix[:500]);shadow.extend(ix[500:1000]);fit.extend(ix[1000:])
 return (np.asarray(sorted(fit),dtype=np.int64),
         np.asarray(sorted(val),dtype=np.int64),
         np.asarray(sorted(shadow),dtype=np.int64))

def luma(rgb):
 r=rgb[...,0].astype(np.uint16);g=rgb[...,1].astype(np.uint16);b=rgb[...,2].astype(np.uint16)
 return ((77*r+150*g+29*b)>>8).astype(np.uint8)

def box_blur(a,r):
 p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
 integ=np.pad(p,((0,0),(1,0),(1,0)),mode="constant")
 integ=integ.cumsum(axis=1,dtype=np.uint32).cumsum(axis=2,dtype=np.uint32)
 k=2*r+1
 s=integ[:,k:,k:]-integ[:,:-k,k:]-integ[:,k:,:-k]+integ[:,:-k,:-k]
 return ((s+(k*k//2))//(k*k)).astype(np.uint8)

def fields(rgb):
 R=rgb[...,0];G=rgb[...,1];B=rgb[...,2]
 Gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
 Y=luma(rgb)
 Chroma=(rgb.max(axis=-1).astype(np.int16)-rgb.min(axis=-1).astype(np.int16)).astype(np.uint8)
 gx=np.zeros_like(Y,dtype=np.int16);gy=np.zeros_like(Y,dtype=np.int16)
 gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
 gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)
 Gx=np.clip(128+gx//2,0,255).astype(np.uint8)
 Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
 Grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)
 lap=np.zeros_like(Y,dtype=np.int16);c=Y[:,1:-1,1:-1].astype(np.int16)
 lap[:,1:-1,1:-1]=(Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c)
 Lap=np.clip(128+lap//4,0,255).astype(np.uint8)
 b1=box_blur(Y,1);b2=box_blur(Y,2);b4=box_blur(Y,4);b8=box_blur(Y,8)
 H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
 H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
 M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
 Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 dgy_dx=np.zeros_like(Y,dtype=np.int16);dgx_dy=np.zeros_like(Y,dtype=np.int16)
 dgy_dx[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2];dgx_dy[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
 Curl=np.clip(128+(dgy_dx-dgx_dy)//4,0,255).astype(np.uint8)
 return {"R":R,"G":G,"B":B,"Gray":Gray,"Luma":Y,"Chroma":Chroma,
         "Gx":Gx,"Gy":Gy,"Grad":Grad,"Laplacian":Lap,"H1":H1,"H2":H2,
         "M4":M4,"L8":L8,"Contrast":Contrast,"Curl":Curl}

def local113(a,bit):
 z=((a>>bit)&1).astype(np.uint8)
 state=z[:,CENTER_ROWS,CENTER_COLS].astype(np.uint16)
 for k,(dr,dc) in enumerate(NEIGHBORS,start=1):
  state |= z[:,CENTER_ROWS+dr,CENTER_COLS+dc].astype(np.uint16)<<k
 return state

def rectangles():
 out=[]
 for r0 in range(0,32,7):
  r1=min(32,r0+7)
  for c0 in range(0,32,16):out.append((r0,r1,c0,c0+16))
 for r0 in range(0,32,16):
  for c0 in range(0,32,7):
   c1=min(32,c0+7);out.append((r0,r0+16,c0,c1))
 assert len(out)==20
 return out
RECTS=rectangles()

def regional(a,bit):
 z=((a>>bit)&1).astype(np.uint8);feats=[]
 for r0,r1,c0,c1 in RECTS:
  b=z[:,r0:r1,c0:c1];count=b.sum(axis=(1,2),dtype=np.uint16)
  xs=np.arange(c1-c0,dtype=np.uint16)[None,None,:];ys=np.arange(r1-r0,dtype=np.uint16)[None,:,None]
  sx=(b.astype(np.uint16)*xs).sum(axis=(1,2),dtype=np.uint32)
  sy=(b.astype(np.uint16)*ys).sum(axis=(1,2),dtype=np.uint32)
  cx=np.full(len(z),255,dtype=np.uint16);cy=np.full(len(z),255,dtype=np.uint16);nz=count>0
  cx[nz]=sx[nz]//count[nz];cy[nz]=sy[nz]//count[nz]
  feats.extend([count.astype(np.uint8),cx.astype(np.uint8),cy.astype(np.uint8)])
 return np.stack(feats,axis=1)

def globalf(a,bit):
 z=((a>>bit)&1).astype(np.uint8);count=z.sum(axis=(1,2),dtype=np.uint16)
 xs=np.arange(32,dtype=np.uint16)[None,None,:];ys=np.arange(32,dtype=np.uint16)[None,:,None]
 sx=(z.astype(np.uint16)*xs).sum(axis=(1,2),dtype=np.uint32);sy=(z.astype(np.uint16)*ys).sum(axis=(1,2),dtype=np.uint32)
 cx=np.full(len(z),255,dtype=np.uint16);cy=np.full(len(z),255,dtype=np.uint16);nz=count>0
 cx[nz]=sx[nz]//count[nz];cy[nz]=sy[nz]//count[nz]
 minx=np.where(z,np.arange(32,dtype=np.uint8)[None,None,:],255).min(axis=(1,2))
 maxx=np.where(z,np.arange(32,dtype=np.uint8)[None,None,:],0).max(axis=(1,2))
 miny=np.where(z,np.arange(32,dtype=np.uint8)[None,:,None],255).min(axis=(1,2))
 maxy=np.where(z,np.arange(32,dtype=np.uint8)[None,:,None],0).max(axis=(1,2))
 minx[~nz]=maxx[~nz]=miny[~nz]=maxy[~nz]=255
 return np.stack([np.minimum(255,count>>2).astype(np.uint8),cx.astype(np.uint8),cy.astype(np.uint8),minx,maxx,miny,maxy],axis=1)

def fit_table(A,y,states):
 F=A.shape[1];cnt=np.zeros((F,K,states),dtype=np.int32)
 off=(states*np.arange(F,dtype=np.int64))[None,:]
 for c in range(K):
  R=A[y==c].astype(np.int64,copy=False)
  bc=np.bincount((R+off).ravel(),minlength=F*states).reshape(F,states)
  cnt[:,c,:]=bc
 total=cnt.sum(axis=1,keepdims=True)
 # Historical v0 log-evidence rule retained exactly for A0 comparability.
 logp=np.log((cnt+ALPHA)/(total+K*ALPHA))
 return np.rint(logp*LOG_SCALE).astype(np.int16).transpose(0,2,1)

def add(tab,A,S):
 F=A.shape[1];fi=np.arange(F,dtype=np.int64)[None,:]
 for st in range(0,len(A),500):
  en=min(len(A),st+500);S[st:en]+=tab[fi,A[st:en]].sum(axis=1,dtype=np.int64)

def acc(S,y):return float(np.mean(S.argmax(axis=1)==y))

def choose(L,R,G,y):
 best=None
 for wl in (0,1,2,4):
  for wr in (0,1,2,4):
   for wg in (0,1,2,4):
    if wl==wr==wg==0:continue
    a=acc(wl*L+wr*R+wg*G,y);key=(a,-wl-wr-wg,-wl,-wr,-wg)
    if best is None or key>best[0]:best=(key,(wl,wr,wg),a)
 return best[1],best[2]

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=split_40_5_5(train_y)
yfit=train_y[fit_idx];yval=train_y[val_idx];yshadow=train_y[shadow_idx]
trf=fields(train_x);tef=fields(test_x)

VL=np.zeros((5000,K),dtype=np.int64);VR=np.zeros_like(VL);VG=np.zeros_like(VL)
SL=np.zeros((5000,K),dtype=np.int64);SR=np.zeros_like(SL);SG=np.zeros_like(SL)
TL=np.zeros((10000,K),dtype=np.int64);TR=np.zeros_like(TL);TG=np.zeros_like(TL)
progress={};model_bytes=0

for ci,name in enumerate(CHANNEL_NAMES):
 a=trf[name];t=tef[name]
 for bit in range(8):
  lf=local113(a,bit);ltf=local113(t,bit)
  tab=fit_table(lf[fit_idx],yfit,512);model_bytes+=tab.nbytes
  add(tab,lf[val_idx],VL);add(tab,lf[shadow_idx],SL);add(tab,ltf,TL)

  rf=regional(a,bit);rt=regional(t,bit)
  tab=fit_table(rf[fit_idx],yfit,256);model_bytes+=tab.nbytes
  add(tab,rf[val_idx],VR);add(tab,rf[shadow_idx],SR);add(tab,rt,TR)

  gf=globalf(a,bit);gt=globalf(t,bit)
  tab=fit_table(gf[fit_idx],yfit,256);model_bytes+=tab.nbytes
  add(tab,gf[val_idx],VG);add(tab,gf[shadow_idx],SG);add(tab,gt,TG)

 w,va=choose(VL,VR,VG,yval)
 progress[name]={"channels_used":ci+1,"weights":list(w),"validation_accuracy":va}
 print(name,progress[name],flush=True)

weights,val_acc=choose(VL,VR,VG,yval);wl,wr,wg=weights
shadow_acc=acc(wl*SL+wr*SR+wg*SG,yshadow)
test_acc=acc(wl*TL+wr*TR+wg*TG,test_y)
passed=shadow_acc>TARGET

R={
 "model":"MPRC-CIFAR-A0-capacity-aligned-113x16",
 "status":"PASS_A0" if passed else "FAIL_A0",
 "target":{"historical_test_baseline":TARGET,"promotion_rule":"shadow > 46.68%"},
 "observation":{
   "stored_byte_samples":1808,
   "shape":[16,113],
   "storage_bits":14464,
   "spatial_anchors":113,
   "channel_names":CHANNEL_NAMES,
   "bitplanes_derived_not_stored":True,
   "anchor_rule":"7*t mod900 mapped to 30x30 valid 3x3-center domain"
 },
 "validation":{"accuracy":val_acc,"selected_weights":list(weights),"channel_progression":progress},
 "shadow_holdout":{"accuracy":shadow_acc,"used_for_selection":False},
 "official_test":{"accuracy":test_acc,"status":"exploratory; historical test exposed"},
 "model_table_bytes":int(model_bytes),
 "runtime_seconds":time.time()-t0,
 "discipline":{"softmax":False,"gradient_descent":False,"resize":False},
 "claim_boundary":"A0 tests whether capacity-aligned distributed evidence can beat the historical v0 classifier. Passing A0 does not imply the >94% goal is reached."
}
out=ROOT/"results"/"cifar_a0_capacity113.json";out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2),flush=True)
if not passed:
 print("A0_REJECT: shadow accuracy did not beat 46.68%",flush=True)
