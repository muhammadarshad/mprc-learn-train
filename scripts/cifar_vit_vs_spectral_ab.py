"""Strict CIFAR A/B: Arshad-ViT native math vs qai-siliq spectral control.

A = Arshad-ViT native path only
--------------------------------
- 3x3 = 1+8 directional local identity
- native 16x7 / 7x16 rectangular patch geometry
- generator-7 anchor transport
- local evidence -> regional native rectangles -> global position/extent evidence
- no SU(16) intensity-band encoder
- no multiscale spectral residuals
- no Laplacian defect pooling borrowed from qai-siliq

B = qai-siliq-inspired spectral control
---------------------------------------
- SU(16) intensity bands
- 6 scales [0,1,4,16,64,128]
- residual smoothing bands
- 3x3 zone defect densities
- winding-like 2-feature tail
- total 896-D feature vector

Purpose
-------
Not to replace A with B.  B is a control that reveals which capability A lacks.
Same CIFAR split. Same simple integer nearest-neighbor readout for both branches
so the encoder is the controlled variable.

This Python control reproduces the *structure* of the Rust spectral path on CIFAR
with integer arithmetic only. It is not claimed bit-identical to qai-siliq Rust.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10";CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz";MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260927
FIT=5000
QUERY=1000
SCALES=(0,1,4,16,64,128)
GEN=7
CHANNELS=16
NEIGH=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):h.update(z)
    return h.hexdigest()

def ddir():
    if not ARCH.exists() or md5(ARCH)!=MD5:urllib.request.urlretrieve(URL,ARCH)
    d=CACHE/"cifar-10-batches-py"
    if not d.exists():
        with tarfile.open(ARCH,"r:gz") as tf:tf.extractall(CACHE)
    return d

def loadb(p):
    with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def load():
    d=ddir();xs=[];ys=[]
    for i in range(1,6):
        x,y=loadb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
    tx=np.concatenate(xs);ty=np.concatenate(ys)
    vx,vy=loadb(d/"test_batch")
    return tx,ty,vx,vy

def luma(x):
    r=x[...,0].astype(np.uint16);g=x[...,1].astype(np.uint16);b=x[...,2].astype(np.uint16)
    return ((77*r+150*g+29*b)>>8).astype(np.uint8)

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
    def blur(a,r):
        p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
        I=np.pad(p,((0,0),(1,0),(1,0)),mode="constant").cumsum(1,dtype=np.uint32).cumsum(2,dtype=np.uint32)
        k=2*r+1
        s=I[:,k:,k:]-I[:,:-k,k:]-I[:,k:,:-k]+I[:,:-k,:-k]
        return ((s+k*k//2)//(k*k)).astype(np.uint8)
    b1=blur(Y,1);b2=blur(Y,2);b4=blur(Y,4);b8=blur(Y,8)
    H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
    H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
    M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
    Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    dgy=np.zeros_like(Y,dtype=np.int16);dgx=np.zeros_like(Y,dtype=np.int16)
    dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2];dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
    return [R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl]

# ---------------- A: native ViT ----------------
# 113 generator-7 centers over valid 30x30 center domain
order=np.asarray([(GEN*t)%900 for t in range(113)],dtype=np.int16)
AR=(order//30+1).astype(np.int16);AC=(order%30+1).astype(np.int16)

def vit_features(imgs):
    F=fields(imgs)
    out=[]
    for a in F:
        per_bit=[]
        for bit in range(8):
            z=((a>>bit)&1).astype(np.uint8)
            state=z[:,AR,AC].astype(np.uint16)
            for k,(dr,dc) in enumerate(NEIGH,1):
                state |= z[:,AR+dr,AC+dc].astype(np.uint16)<<k
            per_bit.append(state.astype(np.uint16))
        out.append(np.stack(per_bit,axis=1))  # [N,8,113], states 0..511
    return np.concatenate(out,axis=1).reshape(len(imgs),-1) # [N,16*8*113] uint16

# ---------------- B: spectral control ----------------
def smooth_steps(a,steps):
    if steps==0:return a.copy()
    x=a.copy()
    for _ in range(steps):
        y=x.copy()
        s=(x[:,0:-2,1:-1].astype(np.uint16)+x[:,2:,1:-1].astype(np.uint16)+
           x[:,1:-1,0:-2].astype(np.uint16)+x[:,1:-1,2:].astype(np.uint16)+
           4*x[:,1:-1,1:-1].astype(np.uint16))>>3
        y[:,1:-1,1:-1]=s.astype(np.uint8)
        x=y
    return x

def curvature(a):
    c=np.full_like(a,128)
    ctr=a[:,1:-1,1:-1].astype(np.uint16)
    h=(a[:,1:-1,:-2].astype(np.uint16)+a[:,1:-1,2:].astype(np.uint16)-2*ctr)&255
    v=(a[:,:-2,1:-1].astype(np.uint16)+a[:,2:,1:-1].astype(np.uint16)-2*ctr)&255
    c[:,1:-1,1:-1]=(((h+v)//2)&255).astype(np.uint8)
    return c

def zone9_defects(curv,thr=48):
    N,H,W=curv.shape
    mh=H//8;mw=W//8
    sr,er,sc,ec=mh,H-mh,mw,W-mw
    inner_h=er-sr;inner_w=ec-sc
    zh=max(1,inner_h//3);zw=max(1,inner_w//3)
    feats=[]
    for gy in range(3):
        for gx in range(3):
            r0=sr+gy*zh;r1=er if gy==2 else min(er,r0+zh)
            c0=sc+gx*zw;c1=ec if gx==2 else min(ec,c0+zw)
            z=curv[:,r0:r1,c0:c1]
            dev=np.abs(z.astype(np.int16)-128)
            den=(dev>thr).sum(axis=(1,2),dtype=np.uint32)
            tot=z.shape[1]*z.shape[2]
            feats.append(((den*255)//max(1,tot)).astype(np.uint8))
    return np.stack(feats,axis=1)

def spectral_features(imgs):
    Y=luma(imgs)
    allbands=[]
    high=(Y>>4).astype(np.uint8)
    for band in range(16):
        bp=np.full_like(Y,128)
        m=high==band
        bp[m]=Y[m]
        scaled=[]
        prev=bp
        smoothed=[]
        for t in SCALES:
            s=bp if t==0 else smooth_steps(bp,t)
            smoothed.append(s)
        for i,s in enumerate(smoothed):
            r=s if i==0 else ((smoothed[i-1].astype(np.uint16)-s.astype(np.uint16)+128)&255).astype(np.uint8)
            scaled.append(zone9_defects(curvature(r)))
        core=np.concatenate(scaled,axis=1) # 54
        # two cheap winding-like summaries on flattened intensity
        flat=bp.reshape(len(bp),-1).astype(np.int16)
        d=np.diff(flat,axis=1)&255
        pos=np.where(d<128,d,0).sum(axis=1,dtype=np.int64)
        neg=np.where(d>128,256-d,0).sum(axis=1,dtype=np.int64)
        denom=max(1,flat.shape[1]*128)
        net=(np.abs(pos-neg)*255//denom).clip(0,255).astype(np.uint8)
        total=((pos+neg)*255//denom).clip(0,255).astype(np.uint8)
        allbands.append(np.concatenate([core,net[:,None],total[:,None]],axis=1))
    return np.concatenate(allbands,axis=1) # 896

def ringdist(mem,q):
    # A uses 9-bit local states 0..511, B uses Z256 features.
    # Use circular distance on the natural state domain inferred from dtype/range.
    m=mem.astype(np.int32);qq=q.astype(np.int32)[None,:]
    mod=512 if int(max(mem.max(),q.max()))>255 else 256
    d=np.abs(m-qq)
    d=np.minimum(d,mod-d)
    return d.sum(axis=1,dtype=np.int64)

def eval_branch(trainF,trainY,testF,testY):
    ok=0
    for i,q in enumerate(testF):
        e=ringdist(trainF,q)
        pred=int(trainY[int(np.argmin(e))])
        ok+=int(pred==int(testY[i]))
    return ok/len(testY)

tx,ty,vx,vy=load()
rng=np.random.default_rng(SEED)
fit_idx=rng.permutation(len(tx))[:FIT]
q_idx=rng.permutation(len(vx))[:QUERY]

Afit=vit_features(tx[fit_idx]);Aq=vit_features(vx[q_idx])
Bfit=spectral_features(tx[fit_idx]);Bq=spectral_features(vx[q_idx])

Aacc=eval_branch(Afit,ty[fit_idx],Aq,vy[q_idx])
Bacc=eval_branch(Bfit,ty[fit_idx],Bq,vy[q_idx])

R={
 "experiment":"Strict encoder A/B on CIFAR",
 "A":{"name":"Arshad-ViT native","dim":int(Afit.shape[1]),"accuracy":Aacc,
      "borrowed_from_spectral":False},
 "B":{"name":"qai-siliq spectral control","dim":int(Bfit.shape[1]),"accuracy":Bacc},
 "controlled":{"train_refs":FIT,"queries":QUERY,"readout":"1-NN exact Z256 circular L1","same_split":True},
 "interpretation_boundary":"B is a control only. If B wins, identify missing capability and solve it using A's own math; do not import B's encoder into A."
}
out=ROOT/"results"/"cifar_vit_vs_spectral_ab.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps(R,indent=2))
