"""CIFAR information-loss localization audit.

No architecture change. Diagnose why accuracy stopped improving.

Tests on the SAME frozen split:
A. v0-style distributed evidence with RGB only vs all 16 deterministic channels.
B. branch contribution: LOCAL vs REGIONAL vs GLOBAL.
C. directional ADI9 factorized byte evidence vs same-information RAW9 factorized evidence.
D. exact joint ADI9 descriptor coverage/purity.

Interpretation:
- A gap RGB->16 means observation/projection richness is a bottleneck.
- RAW9 ~= ADI9 but both weak means the factorized head, not ADI invertibility, loses relation information.
- high purity + low exact coverage means exact SELECT is too sparse.
- regional/global gains identify missing hierarchy rather than local algebra failure.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
K=10; SEED=20260924
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]
CENTER_ROWS=np.arange(2,30,4,dtype=np.int16); CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
ARMS=[(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2)]

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def ensure():
 if not ARCH.exists() or md5(ARCH)!=MD5: urllib.request.urlretrieve(URL,ARCH)
 d=CACHE/"cifar-10-batches-py"
 if not d.exists():
  with tarfile.open(ARCH,"r:gz") as tf: tf.extractall(CACHE)
 return d
def lb(p):
 with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
 return np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy(),np.asarray(d[b"labels"],dtype=np.int64)
def load():
 d=ensure();xs=[];ys=[]
 for i in range(1,6):
  x,y=lb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
 return np.concatenate(xs),np.concatenate(ys)
def split(y):
 rng=np.random.default_rng(SEED);fit=[];val=[]
 for c in range(K):
  ix=np.flatnonzero(y==c);ix=ix[rng.permutation(len(ix))]
  val.extend(ix[:500]);fit.extend(ix[1000:]) # match 40k discipline
 return np.asarray(sorted(fit)),np.asarray(sorted(val))
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
 H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
 H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
 M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
 Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 dgy=np.zeros_like(Y,dtype=np.int16);dgx=np.zeros_like(Y,dtype=np.int16)
 dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2];dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
 Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
 return dict(zip(CHANNEL_NAMES,[R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl]))

def evidence(A,y):
 F=A.shape[1];cnt=np.zeros((F,K,256),dtype=np.int32);off=(256*np.arange(F,dtype=np.int64))[None,:]
 for c in range(K):
  z=A[y==c].astype(np.int64,copy=False)
  cnt[:,c,:]=np.bincount((z+off).ravel(),minlength=F*256).reshape(F,256)
 total=cnt.sum(1,keepdims=True)
 return (K*cnt-total).astype(np.int16).transpose(0,2,1)
def score(T,A):
 fi=np.arange(A.shape[1],dtype=np.int64)[None,:];S=np.zeros((len(A),K),dtype=np.int64)
 for st in range(0,len(A),500): S[st:st+500]=T[fi,A[st:st+500]].sum(1,dtype=np.int64)
 return S
def acc(S,y): return float(np.mean(S.argmax(1)==y))

def raw9(a):
 rr=CENTER_ROWS[:,None];cc=CENTER_COLS[None,:];parts=[a[:,rr,cc]]
 for dr,dc in ARMS:parts.append(a[:,rr+dr,cc+dc])
 return np.stack(parts,-1).reshape(len(a),49,9)
def adi9(a):
 r=raw9(a).astype(np.uint16);C=r[:,:,0];o=np.empty_like(r,dtype=np.uint8)
 o[:,:,0]=(r.sum(2)&255).astype(np.uint8)
 for j in range(1,9):o[:,:,j]=((C-r[:,:,j])&255).astype(np.uint8)
 return o

# v0 local bit-mask representation only; enough to test feature-bank value.
NEI=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
def v0_local(a,bit):
 z=((a>>bit)&1).astype(np.uint8);rr=CENTER_ROWS[:,None];cc=CENTER_COLS[None,:]
 center=z[:,rr,cc].reshape(len(z),49)
 mask=np.zeros((len(z),7,7),dtype=np.uint8)
 for k,(dr,dc) in enumerate(NEI): mask|=z[:,rr+dr,cc+dc]<<k
 return np.concatenate([mask.reshape(len(z),49),center],1)

# simple global stats from v0, to test hierarchy contribution.
def global7(a,bit):
 z=((a>>bit)&1).astype(np.uint8);cnt=z.sum((1,2),dtype=np.uint16)
 xs=np.arange(32,dtype=np.uint16)[None,None,:];ys=np.arange(32,dtype=np.uint16)[None,:,None]
 sx=(z.astype(np.uint16)*xs).sum((1,2),dtype=np.uint32);sy=(z.astype(np.uint16)*ys).sum((1,2),dtype=np.uint32)
 cx=np.full(len(z),255,dtype=np.uint16);cy=cx.copy();nz=cnt>0;cx[nz]=sx[nz]//cnt[nz];cy[nz]=sy[nz]//cnt[nz]
 minx=np.where(z,np.arange(32,dtype=np.uint8)[None,None,:],255).min((1,2));maxx=np.where(z,np.arange(32,dtype=np.uint8)[None,None,:],0).max((1,2))
 miny=np.where(z,np.arange(32,dtype=np.uint8)[None,:,None],255).min((1,2));maxy=np.where(z,np.arange(32,dtype=np.uint8)[None,:,None],0).max((1,2))
 minx[~nz]=255;maxx[~nz]=255;miny[~nz]=255;maxy[~nz]=255
 return np.stack([np.minimum(255,cnt>>2).astype(np.uint8),cx.astype(np.uint8),cy.astype(np.uint8),minx,maxx,miny,maxy],1)

X,y=load();fit,val=split(y);F=fields(X);yf=y[fit];yv=y[val]
R={"experiment":"CIFAR information-loss localization","fit":len(fit),"validation":len(val),"tests":{}}

def run_feature_set(names):
 Sl=np.zeros((len(val),K),dtype=np.int64);Sg=np.zeros_like(Sl);Sa=np.zeros_like(Sl);Sr=np.zeros_like(Sl)
 for name in names:
  a=F[name]
  # v0-style local + global
  for bit in range(8):
   L=v0_local(a,bit);T=evidence(L[fit],yf);Sl+=score(T,L[val])
   G=global7(a,bit);Tg=evidence(G[fit],yf);Sg+=score(Tg,G[val])
  A=adi9(a).reshape(len(a),49*9);RA=raw9(a).reshape(len(a),49*9)
  Sa+=score(evidence(A[fit],yf),A[val]);Sr+=score(evidence(RA[fit],yf),RA[val])
 return {"v0_local":acc(Sl,yv),"v0_global":acc(Sg,yv),"v0_local_plus_global":acc(Sl+Sg,yv),
         "adi9_factorized":acc(Sa,yv),"raw9_factorized":acc(Sr,yv)}

R["tests"]["RGB_only"]=run_feature_set(["R","G","B"])
R["tests"]["all16"]=run_feature_set(CHANNEL_NAMES)

# Exact joint descriptor support + purity on RGB. Sample 1000 validation for runtime.
mem=np.concatenate([adi9(F[n]) for n in ["R","G","B"]],axis=1) # [N,147,9]
VOID=np.dtype((np.void,9)); keys=np.ascontiguousarray(mem[fit]).reshape(-1,9).view(VOID).reshape(-1)
labs=np.repeat(yf,147)
order=np.argsort(keys,kind="stable");keys=keys[order];labs=labs[order]
covered=correct_majority=ambiguous=0; matches=[]
for qi in val[:1000]:
 q=np.ascontiguousarray(mem[qi]).reshape(-1,9).view(VOID).reshape(-1); found=[]
 for key in np.unique(q):
  lo=np.searchsorted(keys,key,"left");hi=np.searchsorted(keys,key,"right")
  if hi>lo: found.extend(labs[lo:hi].tolist())
 if found:
  covered+=1;bc=np.bincount(found,minlength=10);mx=bc.max();w=np.flatnonzero(bc==mx)
  ambiguous+=int(len(w)>1);correct_majority+=int(len(w)==1 and int(w[0])==int(y[qi]));matches.append(len(found))
R["tests"]["exact_joint_ADI9_RGB"]={"queries":1000,"coverage":covered/1000,"majority_accuracy_all":correct_majority/1000,
 "majority_accuracy_covered":correct_majority/covered if covered else None,"ambiguous_majority":ambiguous,
 "mean_occurrence_hits":float(np.mean(matches)) if matches else 0.0}

# Derived diagnosis, mechanically based on measured gaps.
rgb=R["tests"]["RGB_only"];all16=R["tests"]["all16"];joint=R["tests"]["exact_joint_ADI9_RGB"]
R["diagnosis"]={
 "projection_gain_local":all16["v0_local"]-rgb["v0_local"],
 "projection_gain_local_global":all16["v0_local_plus_global"]-rgb["v0_local_plus_global"],
 "hierarchy_gain_all16":all16["v0_local_plus_global"]-all16["v0_local"],
 "adi_vs_raw9_gap":all16["adi9_factorized"]-all16["raw9_factorized"],
 "factorized_vs_v0local_gap":all16["adi9_factorized"]-all16["v0_local"],
 "exact_joint_coverage":joint["coverage"]
}
out=ROOT/"results"/"cifar_information_loss_localization.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps(R,indent=2),flush=True)
