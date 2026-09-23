"""
MPRC-Vision-CIFAR geometry A/B
====================

First end-to-end MPRC-family CIFAR-10 image classifier.

Architecture frozen before official-test scoring:
  16 deterministic integer channels
  x 8 bit planes
  x three observer levels

LOCAL:
  Corrected MPRC 1+8 directional context:
    C + (U1,U2) + (D1,D2) + (F1,F2) + (B1,B2).
  This is the two-depth 5-arm cross. The nine binary observations are
  kept jointly as one 9-bit directional shape state at each anchor.

REGIONAL:
  Native 7x16 and 16x7 rectangles.  For every channel/bit plane/block,
  observe active-bit count and active-pixel centroid (local displacement).

GLOBAL:
  Per channel/bit plane active count, centroid, and bounding extent.

LEARN:
  Observation-populated categorical evidence tables.
  No gradient descent, QKV, softmax, convolution weights, or GPU training.
  Tables are built on 40k train-fit images. A 5k validation set chooses
  branch loudnesses; a separate 5k shadow holdout from the official training
  partition is untouched by those choices. The official CIFAR test is an
  exploratory continuity benchmark because v0 already exposed it.

This is a real classifier but not yet full Arshad-ViT B5:
  task-specific REACT-LUT learning remains an explicitly open seam in the
  frozen reference architecture.  This model is MPRC IDENTIFY/shape geometry
  + native rectangular transport + MPRC-Learn evidence head.
"""

from pathlib import Path
import hashlib, json, pickle, tarfile, urllib.request, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"
CACHE.mkdir(parents=True,exist_ok=True)

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"

K=10
ALPHA=0.05
LOG_SCALE=16
SEED=20260924

CHANNEL_NAMES=[
    "R","G","B","Gray","Luma","Chroma","Gx","Gy",
    "Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"
]

CENTER_ROWS=np.arange(2,30,4,dtype=np.int16)
CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
assert len(CENTER_ROWS)==7 and len(CENTER_COLS)==7

DIRECTIONAL_ARMS=[
    (-1,0),(-2,0),
    (1,0),(2,0),
    (0,1),(0,2),
    (0,-1),(0,-2),
]

def md5(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dataset():
    if not ARCHIVE.exists() or md5(ARCHIVE)!=MD5:
        urllib.request.urlretrieve(URL,ARCHIVE)
    assert md5(ARCHIVE)==MD5
    folder=CACHE/"cifar-10-batches-py"
    if not folder.exists():
        with tarfile.open(ARCHIVE,"r:gz") as tf:
            tf.extractall(CACHE)
    return folder

def load_batch(path):
    with open(path,"rb") as f:
        d=pickle.load(f,encoding="bytes")
    data=d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.uint8)
    labels=np.asarray(d[b"labels"],dtype=np.int64)
    return data,labels

def load_cifar():
    folder=ensure_dataset()
    xs=[];ys=[]
    for i in range(1,6):
        x,y=load_batch(folder/f"data_batch_{i}")
        xs.append(x);ys.append(y)
    train_x=np.concatenate(xs);train_y=np.concatenate(ys)
    test_x,test_y=load_batch(folder/"test_batch")
    return train_x,train_y,test_x,test_y

def stratified_fit_val_shadow(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[];shadow=[]
    for c in range(K):
        ix=np.flatnonzero(y==c)
        ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500].tolist())
        shadow.extend(ix[500:1000].tolist())
        fit.extend(ix[1000:].tolist())
    fit=np.asarray(sorted(fit),dtype=np.int64)
    val=np.asarray(sorted(val),dtype=np.int64)
    shadow=np.asarray(sorted(shadow),dtype=np.int64)
    assert len(fit)==40000 and len(val)==5000 and len(shadow)==5000
    return fit,val,shadow

def luma(rgb):
    r=rgb[...,0].astype(np.uint16)
    g=rgb[...,1].astype(np.uint16)
    b=rgb[...,2].astype(np.uint16)
    return ((77*r+150*g+29*b)>>8).astype(np.uint8)

def box_blur(a,r):
    # Integral-image box blur, edge-replicated. Integer only.
    p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
    integ=np.pad(p,((0,0),(1,0),(1,0)),mode="constant")
    integ=integ.cumsum(axis=1,dtype=np.uint32).cumsum(axis=2,dtype=np.uint32)
    k=2*r+1
    s=(integ[:,k:,k:]-integ[:,:-k,k:]-integ[:,k:,:-k]+integ[:,:-k,:-k])
    return ((s+(k*k//2))//(k*k)).astype(np.uint8)

def base_fields(rgb):
    R=rgb[...,0]
    G=rgb[...,1]
    B=rgb[...,2]
    Gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
    Y=luma(rgb)
    Chroma=(rgb.max(axis=-1).astype(np.int16)-rgb.min(axis=-1).astype(np.int16)).astype(np.uint8)

    gx=np.zeros_like(Y,dtype=np.int16)
    gy=np.zeros_like(Y,dtype=np.int16)
    gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
    gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)

    Gx=np.clip(128+gx//2,0,255).astype(np.uint8)
    Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
    Grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)

    lap=np.zeros_like(Y,dtype=np.int16)
    c=Y[:,1:-1,1:-1].astype(np.int16)
    lap[:,1:-1,1:-1]=(
        Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+
        Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c
    )
    Lap=np.clip(128+lap//4,0,255).astype(np.uint8)

    b1=box_blur(Y,1)
    b2=box_blur(Y,2)
    b4=box_blur(Y,4)
    b8=box_blur(Y,8)
    H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
    H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
    M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
    Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)

    # curl-like integer cross-derivative response, centered at 128.
    dgy_dx=np.zeros_like(Y,dtype=np.int16)
    dgx_dy=np.zeros_like(Y,dtype=np.int16)
    dgy_dx[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2]
    dgx_dy[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy_dx-dgx_dy)//4,0,255).astype(np.uint8)

    return {
        "R":R,"G":G,"B":B,"Gray":Gray,"Luma":Y,"Chroma":Chroma,
        "Gx":Gx,"Gy":Gy,"Grad":Grad,"Laplacian":Lap,
        "H1":H1,"H2":H2,"M4":M4,"L8":L8,
        "Contrast":Contrast,"Curl":Curl,
    }

def rectangles():
    out=[]
    for r0 in range(0,32,7):
        r1=min(32,r0+7)
        for c0 in range(0,32,16):
            out.append(("H",r0,r1,c0,c0+16))
    for r0 in range(0,32,16):
        for c0 in range(0,32,7):
            c1=min(32,c0+7)
            out.append(("V",r0,r0+16,c0,c1))
    assert len(out)==20
    return out

RECTS=rectangles()

SQUARE_NEIGHBORS=[
    (-1,-1),(-1,0),(-1,1),(0,-1),
    (0,1),(1,-1),(1,0),(1,1)
]

def local_features_directional(a,bit):
    """B: C + 2U + 2D + 2F + 2B."""
    z=((a>>bit)&1).astype(np.uint8)
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    state=z[:,rr,cc].astype(np.uint16)
    for k,(dr,dc) in enumerate(DIRECTIONAL_ARMS, start=1):
        state |= (z[:,rr+dr,cc+dc].astype(np.uint16)<<k)
    return state.reshape(len(z),49)

def local_features_square(a,bit):
    """A: matched joint 9-bit center + eight immediate square neighbours."""
    z=((a>>bit)&1).astype(np.uint8)
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    state=z[:,rr,cc].astype(np.uint16)
    for k,(dr,dc) in enumerate(SQUARE_NEIGHBORS, start=1):
        state |= (z[:,rr+dr,cc+dc].astype(np.uint16)<<k)
    return state.reshape(len(z),49)

def regional_features(a,bit):
    z=((a>>bit)&1).astype(np.uint8)
    feats=[]
    for orient,r0,r1,c0,c1 in RECTS:
        b=z[:,r0:r1,c0:c1]
        count=b.sum(axis=(1,2),dtype=np.uint16)

        xs=np.arange(c1-c0,dtype=np.uint16)[None,None,:]
        ys=np.arange(r1-r0,dtype=np.uint16)[None,:,None]
        sx=(b.astype(np.uint16)*xs).sum(axis=(1,2),dtype=np.uint32)
        sy=(b.astype(np.uint16)*ys).sum(axis=(1,2),dtype=np.uint32)

        cx=np.full(len(z),255,dtype=np.uint16)
        cy=np.full(len(z),255,dtype=np.uint16)
        nz=count>0
        cx[nz]=sx[nz]//count[nz]
        cy[nz]=sy[nz]//count[nz]

        feats.extend([count.astype(np.uint8),cx.astype(np.uint8),cy.astype(np.uint8)])
    return np.stack(feats,axis=1)

def global_features(a,bit):
    z=((a>>bit)&1).astype(np.uint8)
    count=z.sum(axis=(1,2),dtype=np.uint16)
    xs=np.arange(32,dtype=np.uint16)[None,None,:]
    ys=np.arange(32,dtype=np.uint16)[None,:,None]
    sx=(z.astype(np.uint16)*xs).sum(axis=(1,2),dtype=np.uint32)
    sy=(z.astype(np.uint16)*ys).sum(axis=(1,2),dtype=np.uint32)

    cx=np.full(len(z),255,dtype=np.uint16)
    cy=np.full(len(z),255,dtype=np.uint16)
    nz=count>0
    cx[nz]=sx[nz]//count[nz]
    cy[nz]=sy[nz]//count[nz]

    minx=np.where(z, np.arange(32,dtype=np.uint8)[None,None,:], 255).min(axis=(1,2))
    maxx=np.where(z, np.arange(32,dtype=np.uint8)[None,None,:], 0).max(axis=(1,2))
    miny=np.where(z, np.arange(32,dtype=np.uint8)[None,:,None], 255).min(axis=(1,2))
    maxy=np.where(z, np.arange(32,dtype=np.uint8)[None,:,None], 0).max(axis=(1,2))
    minx[~nz]=255;maxx[~nz]=255;miny[~nz]=255;maxy[~nz]=255

    count_bin=np.minimum(255,count>>2).astype(np.uint8)
    return np.stack([
        count_bin,cx.astype(np.uint8),cy.astype(np.uint8),
        minx,maxx,miny,maxy
    ],axis=1)

def fit_table(A,yfit,states=256):
    F=A.shape[1]
    cnt=np.zeros((F,K,states),dtype=np.int32)
    off=(states*np.arange(F,dtype=np.int64))[None,:]
    for c in range(K):
        R=A[yfit==c].astype(np.int64,copy=False)
        codes=R+off
        bc=np.bincount(codes.ravel(),minlength=F*states).reshape(F,states)
        cnt[:,c,:]=bc
    total=cnt.sum(axis=1,keepdims=True)
    logp=np.log((cnt+ALPHA)/(total+K*ALPHA))
    return np.rint(logp*LOG_SCALE).astype(np.int16).transpose(0,2,1)

def add_scores(table,A,S):
    F=A.shape[1]
    fi=np.arange(F,dtype=np.int64)[None,:]
    CHUNK=500
    for start in range(0,len(A),CHUNK):
        stop=min(len(A),start+CHUNK)
        S[start:stop]+=table[fi,A[start:stop]].sum(axis=1,dtype=np.int64)

def branch_fit_score(Atr,ytr,eval_sets,states=256):
    tab=fit_table(Atr,ytr,states)
    for A,S in eval_sets:
        add_scores(tab,A,S)
    return int(tab.nbytes)

def accuracy(S,y):
    return float(np.mean(S.argmax(axis=1)==y))

def best_branch_weights(local,regional,global_,yval):
    choices=[0,1,2,4]
    best=None
    for wl in choices:
        for wr in choices:
            for wg in choices:
                if wl==wr==wg==0:
                    continue
                S=wl*local+wr*regional+wg*global_
                a=accuracy(S,yval)
                key=(a,-(wl+wr+wg),-wl,-wr,-wg)
                if best is None or key>best[0]:
                    best=(key,(wl,wr,wg),a)
    return best[1],best[2]

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=stratified_fit_val_shadow(train_y)
yfit=train_y[fit_idx]
yval=train_y[val_idx]
yshadow=train_y[shadow_idx]

# Classifier branch scores; official test labels are intentionally not consulted
# until branch weights have been frozen from validation.
Vdir=np.zeros((len(val_idx),K),dtype=np.int64)
Vsq=np.zeros_like(Vdir)
Vglobal=np.zeros_like(Vdir)
Sdir=np.zeros((len(shadow_idx),K),dtype=np.int64)
Ssq=np.zeros_like(Sdir)
Sglobal=np.zeros_like(Sdir)
Tdir=np.zeros((len(test_y),K),dtype=np.int64)
Tsq=np.zeros_like(Tdir)
Tglobal=np.zeros_like(Tdir)

train_fields=base_fields(train_x)
test_fields=base_fields(test_x)

model_bytes=0
channel_validation={}

for ci,name in enumerate(CHANNEL_NAMES):
    atr=train_fields[name]
    ate=test_fields[name]

    for bit in range(8):
        ld=local_features_directional(atr,bit)
        ltd=local_features_directional(ate,bit)
        ls=local_features_square(atr,bit)
        lts=local_features_square(ate,bit)

        model_bytes+=branch_fit_score(
            ld[fit_idx],yfit,
            [(ld[val_idx],Vdir),(ld[shadow_idx],Sdir),(ltd,Tdir)],
            512
        )
        model_bytes+=branch_fit_score(
            ls[fit_idx],yfit,
            [(ls[val_idx],Vsq),(ls[shadow_idx],Ssq),(lts,Tsq)],
            512
        )

        gg=global_features(atr,bit)
        gt=global_features(ate,bit)
        model_bytes+=branch_fit_score(
            gg[fit_idx],yfit,
            [(gg[val_idx],Vglobal),(gg[shadow_idx],Sglobal),(gt,Tglobal)],
            256
        )

    # Matched local-only and local+global checks. Global loudness in {0,1,2,4}.
    def choose(local, global_, y):
        best=None
        for wl in (1,2,4):
            for wg in (0,1,2,4):
                a=accuracy(wl*local+wg*global_,y)
                key=(a,-(wl+wg),-wl,-wg)
                if best is None or key>best[0]:
                    best=(key,(wl,wg),a)
        return best[1],best[2]

    dw,dva=choose(Vdir,Vglobal,yval)
    sw,sva=choose(Vsq,Vglobal,yval)
    channel_validation[name]={
        "channels_used":ci+1,
        "directional_weights":list(dw),
        "directional_validation_accuracy":dva,
        "square_weights":list(sw),
        "square_validation_accuracy":sva,
        "directional_minus_square":dva-sva,
    }
    print(name,channel_validation[name],flush=True)

# Freeze each geometry independently on validation, then read shadow.
def choose_final(local,global_,y):
    best=None
    for wl in (1,2,4):
        for wg in (0,1,2,4):
            a=accuracy(wl*local+wg*global_,y)
            key=(a,-(wl+wg),-wl,-wg)
            if best is None or key>best[0]:
                best=(key,(wl,wg),a)
    return best[1],best[2]

(dir_w,dir_val)=choose_final(Vdir,Vglobal,yval)
(sq_w,sq_val)=choose_final(Vsq,Vglobal,yval)
dwl,dwg=dir_w
swl,swg=sq_w

dir_shadow=accuracy(dwl*Sdir+dwg*Sglobal,yshadow)
sq_shadow=accuracy(swl*Ssq+swg*Sglobal,yshadow)
dir_test=accuracy(dwl*Tdir+dwg*Tglobal,test_y)
sq_test=accuracy(swl*Tsq+swg*Tglobal,test_y)

# Confusion only for the directional candidate.
test_scores=dwl*Tdir+dwg*Tglobal
pred=test_scores.argmax(axis=1)
conf=np.zeros((K,K),dtype=np.int64)
for truth,p in zip(test_y,pred):
    conf[int(truth),int(p)]+=1

result={
    "model":"MPRC-Vision-CIFAR-geometry-AB",
    "status":"EMPIRICAL REAL CLASSIFIER; not yet full Arshad-ViT B5",
    "dataset":{
        "train_fit":40000,
        "validation":5000,
        "shadow_holdout":5000,
        "official_test":10000,
        "shape":[32,32,3],
        "archive_md5":MD5,
    },
    "architecture":{
        "channels":CHANNEL_NAMES,
        "bit_planes_per_channel":8,
        "local":{
            "A_square":"C + 8 immediate square neighbours",
            "B_directional":"C + U1,U2 + D1,D2 + F1,F2 + B1,B2",
            "state":"matched joint 9-bit shape",
            "states":512,
            "anchors":"matched 7x7 stride-4 lattice",
            "features_per_channel_bit":49
        },
        "regional":{
            "geometry":"10x 7x16 + 10x 16x7 explicit partial rectangles",
            "observations_per_block":["active_count","centroid_x","centroid_y"],
            "features_per_channel_bit":60,
        },
        "global":{
            "observations":["active_count_bin","centroid_x","centroid_y",
                            "min_x","max_x","min_y","max_y"],
            "features_per_channel_bit":7,
        },
        "learning":"categorical occupancy -> quantized integer log-evidence LUT",
        "softmax":False,
        "gradient_descent":False,
        "gpu_required":False,
    },
    "validation":{
        "channel_progression":channel_validation,
        "directional":{"weights":{"local":dwl,"global":dwg},"accuracy":dir_val},
        "square":{"weights":{"local":swl,"global":swg},"accuracy":sq_val},
        "directional_minus_square":dir_val-sq_val
    },
    "shadow_holdout":{
        "total":int(len(yshadow)),
        "selection_used_shadow_labels":False,
        "directional_accuracy":dir_shadow,
        "square_accuracy":sq_shadow,
        "directional_minus_square":dir_shadow-sq_shadow
    },
    "official_test":{
        "status":"exploratory only; both geometries already downstream of exposed test history",
        "directional_accuracy":dir_test,
        "square_accuracy":sq_test,
        "directional_minus_square":dir_test-sq_test,
        "directional_confusion_matrix":conf.tolist()
    },
    "model_table_bytes":int(model_bytes),
    "runtime_seconds":float(time.time()-t0),
    "controlled_change":"Strict geometry A/B: same 40k/5k/5k split, same 7x7 anchors, same joint 9-bit state, same channels, same evidence rule, same global branch. Only the eight local sample positions differ.",
    "open_seam":"Frozen reference leaves task-specific REACT LUT learning/selection open; v1 therefore uses MPRC-Learn evidence as the classifier head rather than claiming a completed BIND-REACT-MEASURE B5 model."
}

out=ROOT/"results"/"mprc_vision_cifar_geometry_ab.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
