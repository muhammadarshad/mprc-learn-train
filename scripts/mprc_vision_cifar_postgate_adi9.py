"""
MPRC Vision CIFAR — post-gate directional ADI-9 classifier
==========================================================

RESEARCH DISCIPLINE
-------------------
This script is eligible to train only after the independent
directional_adi9_survival_gate.py has passed.  The workflow executes that
gate first.

The local observer is the proved Z256 automorphism

  a = (C,U1,U2,D1,D2,F1,F2,B1,B2)

  Lambda  = sum(a) mod 256
  delta_j = C-a_j mod 256

and ONLY the nine resulting u8 coordinates are supplied to learning.

No 9-bit / 512-state scalar is used.
No gradient descent, softmax, QKV, float probability, or GPU is used.
Learning is observation population:

  E_c(f,s) = K * N_c(f,s) - N(f,s)

with balanced classes (K=10), followed by integer evidence summation.

The 7x7 observer lattice on native 32x32 CIFAR is EXPERIMENTAL sampling
topology; it is not claimed as a frozen Arshad-ViT multires theorem.
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
SEED=20260924
INV9=57

CHANNEL_NAMES=[
    "R","G","B","Gray","Luma","Chroma","Gx","Gy",
    "Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"
]

# Experimental CIFAR32 observer sampling, frozen before labels are read.
CENTER_ROWS=np.arange(2,30,4,dtype=np.int16)
CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
assert len(CENTER_ROWS)==7 and len(CENTER_COLS)==7

ARMS=[
    (-1,0),(-2,0),  # U1,U2
    (1,0),(2,0),    # D1,D2
    (0,1),(0,2),    # F1,F2
    (0,-1),(0,-2),  # B1,B2
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
    x=d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.uint8)
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def load_cifar():
    folder=ensure_dataset()
    xs=[];ys=[]
    for i in range(1,6):
        x,y=load_batch(folder/f"data_batch_{i}")
        xs.append(x);ys.append(y)
    train_x=np.concatenate(xs)
    train_y=np.concatenate(ys)
    test_x,test_y=load_batch(folder/"test_batch")
    return train_x,train_y,test_x,test_y

def split_40_5_5(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[];shadow=[]
    for c in range(K):
        ix=np.flatnonzero(y==c)
        ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500])
        shadow.extend(ix[500:1000])
        fit.extend(ix[1000:])
    fit=np.asarray(sorted(fit),dtype=np.int64)
    val=np.asarray(sorted(val),dtype=np.int64)
    shadow=np.asarray(sorted(shadow),dtype=np.int64)
    assert (len(fit),len(val),len(shadow))==(40000,5000,5000)
    assert all(np.bincount(y[ix],minlength=K).tolist()==([4000]*K if len(ix)==40000 else [500]*K)
               for ix in (fit,val,shadow))
    return fit,val,shadow

def luma(rgb):
    r=rgb[...,0].astype(np.uint16)
    g=rgb[...,1].astype(np.uint16)
    b=rgb[...,2].astype(np.uint16)
    return ((77*r+150*g+29*b)>>8).astype(np.uint8)

def box_blur(a,r):
    p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
    integ=np.pad(p,((0,0),(1,0),(1,0)),mode="constant")
    integ=integ.cumsum(axis=1,dtype=np.uint32).cumsum(axis=2,dtype=np.uint32)
    k=2*r+1
    s=(integ[:,k:,k:]-integ[:,:-k,k:]-integ[:,k:,:-k]+integ[:,:-k,:-k])
    return ((s+(k*k//2))//(k*k)).astype(np.uint8)

def fields(rgb):
    R=rgb[...,0]; G=rgb[...,1]; B=rgb[...,2]
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

    b1=box_blur(Y,1); b2=box_blur(Y,2); b4=box_blur(Y,4); b8=box_blur(Y,8)
    H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
    H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
    M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
    Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)

    dgy_dx=np.zeros_like(Y,dtype=np.int16)
    dgx_dy=np.zeros_like(Y,dtype=np.int16)
    dgy_dx[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2]
    dgx_dy[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy_dx-dgx_dy)//4,0,255).astype(np.uint8)

    return {
        "R":R,"G":G,"B":B,"Gray":Gray,"Luma":Y,"Chroma":Chroma,
        "Gx":Gx,"Gy":Gy,"Grad":Grad,"Laplacian":Lap,
        "H1":H1,"H2":H2,"M4":M4,"L8":L8,"Contrast":Contrast,"Curl":Curl,
    }

def adi9(a):
    """Return [N,49,9] u8: Lambda plus eight center-to-arm deltas."""
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    C=a[:,rr,cc].astype(np.uint16)  # [N,7,7]
    vals=[]
    total=C.copy()
    for dr,dc in ARMS:
        v=a[:,rr+dr,cc+dc].astype(np.uint16)
        vals.append(v)
        total=(total+v)&0xFF
    out=np.empty((len(a),7,7,9),dtype=np.uint8)
    out[:,:,:,0]=total.astype(np.uint8)
    for j,v in enumerate(vals,start=1):
        out[:,:,:,j]=((C-v)&0xFF).astype(np.uint8)
    return out.reshape(len(a),49,9)

def raw9(a):
    """Same information before ADI transform, used only as a factorization control."""
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    parts=[a[:,rr,cc]]
    for dr,dc in ARMS:
        parts.append(a[:,rr+dr,cc+dc])
    return np.stack(parts,axis=-1).reshape(len(a),49,9)

def local_roundtrip_gate(a):
    """Exact code-path gate on sampled real channel bytes before labels are used."""
    q=adi9(a)
    lam=q[...,0].astype(np.uint16)
    ds=q[...,1:].astype(np.uint16)
    C=(INV9*((lam+ds.sum(axis=-1,dtype=np.uint16))&0xFF))&0xFF

    rec=np.empty_like(q)
    rec[...,0]=C.astype(np.uint8)
    for j in range(8):
        rec[...,j+1]=((C-ds[...,j])&0xFF).astype(np.uint8)

    source=raw9(a)
    assert np.array_equal(rec,source)
    return int(source.size)

def fit_evidence(A,y):
    """Integer observation-populated evidence table: E_c=10*N_c-N."""
    F=A.shape[1]
    cnt=np.zeros((F,K,256),dtype=np.int32)
    off=(256*np.arange(F,dtype=np.int64))[None,:]
    for c in range(K):
        R=A[y==c].astype(np.int64,copy=False)
        bc=np.bincount((R+off).ravel(),minlength=F*256).reshape(F,256)
        cnt[:,c,:]=bc
    total=cnt.sum(axis=1,keepdims=True)
    E=K*cnt-total
    # Training counts max at 4000, evidence magnitude is well inside int16.
    assert E.min()>=-32768 and E.max()<=32767
    return E.astype(np.int16).transpose(0,2,1)

def score(tab,A):
    F=A.shape[1]
    fi=np.arange(F,dtype=np.int64)[None,:]
    S=np.zeros((len(A),K),dtype=np.int64)
    CH=500
    for st in range(0,len(A),CH):
        en=min(len(A),st+CH)
        S[st:en]=tab[fi,A[st:en]].sum(axis=1,dtype=np.int64)
    return S

def acc(S,y):
    return float(np.mean(S.argmax(axis=1)==y))

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=split_40_5_5(train_y)

# IMPORTANT: all geometry/code gates above are independent of labels.
train_fields=fields(train_x)
test_fields=fields(test_x)

# Real-data byte round-trip check before evidence learning.
roundtrip_bytes=0
for name in CHANNEL_NAMES:
    # Check 512 deterministic images per channel; theorem gate has already covered
    # the full algebraic domain, this catches integration/indexing mistakes.
    roundtrip_bytes += local_roundtrip_gate(train_fields[name][:512])

yfit=train_y[fit_idx]
yval=train_y[val_idx]
yshadow=train_y[shadow_idx]

Vadi=np.zeros((len(val_idx),K),dtype=np.int64)
Sadi=np.zeros((len(shadow_idx),K),dtype=np.int64)
Tadi=np.zeros((len(test_y),K),dtype=np.int64)

# Same-information raw coordinate control.  It is NOT model selection; both are
# reported to expose whether the ADI coordinate system helps the factorized
# evidence head.
Vraw=np.zeros_like(Vadi)
Sraw=np.zeros_like(Sadi)
Traw=np.zeros_like(Tadi)

model_bytes_adi=0
model_bytes_raw=0
progress={}

for ci,name in enumerate(CHANNEL_NAMES):
    tr=train_fields[name]
    te=test_fields[name]

    qa=adi9(tr).reshape(len(tr),49*9)
    ta=adi9(te).reshape(len(te),49*9)
    qr=raw9(tr).reshape(len(tr),49*9)
    traw=raw9(te).reshape(len(te),49*9)

    taba=fit_evidence(qa[fit_idx],yfit)
    tabr=fit_evidence(qr[fit_idx],yfit)
    model_bytes_adi+=taba.nbytes
    model_bytes_raw+=tabr.nbytes

    Vadi += score(taba,qa[val_idx])
    Sadi += score(taba,qa[shadow_idx])
    Tadi += score(taba,ta)

    Vraw += score(tabr,qr[val_idx])
    Sraw += score(tabr,qr[shadow_idx])
    Traw += score(tabr,traw)

    progress[name]={
        "channels_used":ci+1,
        "adi_validation":acc(Vadi,yval),
        "raw9_validation":acc(Vraw,yval),
    }
    print(name,progress[name],flush=True)

adi_val=acc(Vadi,yval)
adi_shadow=acc(Sadi,yshadow)
adi_test=acc(Tadi,test_y)
raw_val=acc(Vraw,yval)
raw_shadow=acc(Sraw,yshadow)
raw_test=acc(Traw,test_y)

pred=Tadi.argmax(axis=1)
conf=np.zeros((K,K),dtype=np.int64)
for y,p in zip(test_y,pred):
    conf[int(y),int(p)]+=1

result={
    "model":"MPRC-Vision-CIFAR-postgate-ADI9",
    "discipline":{
        "required_survival_gate":"directional_adi9_survival_gate.py",
        "training_workflow_runs_gate_first":True,
        "real_data_prelabel_roundtrip_values_checked":roundtrip_bytes,
        "quarantined_512_state_feature_used":False,
    },
    "dataset":{
        "fit":40000,"validation":5000,"shadow_holdout":5000,
        "official_test":10000,
        "official_test_status":"exploratory; previously exposed by older experiments",
        "archive_md5":MD5,
    },
    "observer":{
        "context":["C","U1","U2","D1","D2","F1","F2","B1","B2"],
        "descriptor":["Lambda","dU1","dU2","dD1","dD2","dF1","dF2","dB1","dB2"],
        "domain":"9 independent u8/Z256 components",
        "anchors":49,
        "sampling_status":"experimental CIFAR32 topology",
    },
    "learning":{
        "rule":"E_c(f,s)=10*N_c(f,s)-N(f,s)",
        "arithmetic":"integer",
        "gradient_descent":False,
        "softmax":False,
        "channel_weights":"all 16 fixed at 1 before training",
    },
    "adi9":{
        "validation_accuracy":adi_val,
        "shadow_accuracy":adi_shadow,
        "exploratory_test_accuracy":adi_test,
        "model_table_bytes":model_bytes_adi,
        "confusion_matrix":conf.tolist(),
    },
    "raw9_same_information_control":{
        "validation_accuracy":raw_val,
        "shadow_accuracy":raw_shadow,
        "exploratory_test_accuracy":raw_test,
        "model_table_bytes":model_bytes_raw,
    },
    "adi9_minus_raw9":{
        "validation":adi_val-raw_val,
        "shadow":adi_shadow-raw_shadow,
        "exploratory_test":adi_test-raw_test,
    },
    "channel_progression_validation_only":progress,
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"Empirical classifier result. ADI9 invertibility is proved/gated; CIFAR usefulness and the experimental 49-anchor sampling are empirical."
}

out=ROOT/"results"/"mprc_vision_cifar_postgate_adi9.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
