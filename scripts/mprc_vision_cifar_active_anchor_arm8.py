"""
MPRC Vision CIFAR — gated active-anchor displacement + Arm8 classifier
======================================================================

PRE-TRAINING GATES REQUIRED BY WORKFLOW:
  1. directional_arm8_w2_survival_gate.py
  2. global_local_displacement_survival_gate.py
  3. active_anchor_arm8_serialization_gate.py

For each channel bit-plane:
  P = active centers on the 7x7 observer lattice
  g = min_lex(P)
  d(p) = p-g
  A(p) = two-depth Arm8 byte from the full-resolution bit-plane

Evidence branches:
  A       : local context only (256 states)
  D       : normalized global displacement only (91 host states)
  D+A     : exact joint host lookup (91*256 states)

D and D+A are metadata/table addresses only, not Z256 arithmetic states.
Only A is a ring byte.

Learning:
  E_c(s) = 10*N_c(s)-N(s), exact integer observation counts.
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
D_STATES=91
A_STATES=256
J_STATES=D_STATES*A_STATES

CHANNEL_NAMES=[
    "R","G","B","Gray","Luma","Chroma","Gx","Gy",
    "Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"
]

CENTER_ROWS=np.arange(2,30,4,dtype=np.int16)
CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
assert len(CENTER_ROWS)==7 and len(CENTER_COLS)==7

ARMS=[
    (-1,0),(-2,0),
    (1,0),(2,0),
    (0,1),(0,2),
    (0,-1),(0,-2),
]

GRID_R=np.repeat(np.arange(7,dtype=np.int16),7)
GRID_C=np.tile(np.arange(7,dtype=np.int16),7)

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
    train_x=np.concatenate(xs);train_y=np.concatenate(ys)
    test_x,test_y=load_batch(folder/"test_batch")
    return train_x,train_y,test_x,test_y

def split_40_5_5(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[];shadow=[]
    for c in range(K):
        ix=np.flatnonzero(y==c)
        ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500]);shadow.extend(ix[500:1000]);fit.extend(ix[1000:])
    return (np.asarray(sorted(fit),dtype=np.int64),
            np.asarray(sorted(val),dtype=np.int64),
            np.asarray(sorted(shadow),dtype=np.int64))

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

    lap=np.zeros_like(Y,dtype=np.int16)
    c=Y[:,1:-1,1:-1].astype(np.int16)
    lap[:,1:-1,1:-1]=(
        Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+
        Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c)
    Lap=np.clip(128+lap//4,0,255).astype(np.uint8)

    b1=box_blur(Y,1);b2=box_blur(Y,2);b4=box_blur(Y,4);b8=box_blur(Y,8)
    H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
    H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
    M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
    Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)

    dgy_dx=np.zeros_like(Y,dtype=np.int16);dgx_dy=np.zeros_like(Y,dtype=np.int16)
    dgy_dx[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2]
    dgx_dy[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy_dx-dgx_dy)//4,0,255).astype(np.uint8)

    return {
        "R":R,"G":G,"B":B,"Gray":Gray,"Luma":Y,"Chroma":Chroma,
        "Gx":Gx,"Gy":Gy,"Grad":Grad,"Laplacian":Lap,
        "H1":H1,"H2":H2,"M4":M4,"L8":L8,
        "Contrast":Contrast,"Curl":Curl,
    }

def observations(a,bit):
    """Return A,D,J arrays [N,49], inactive entries = -1."""
    z=((a>>bit)&1).astype(np.uint8)
    rr=CENTER_ROWS[:,None];cc=CENTER_COLS[None,:]

    active=z[:,rr,cc].reshape(len(z),49).astype(bool)

    A=np.zeros((len(z),7,7),dtype=np.uint16)
    for k,(dr,dc) in enumerate(ARMS):
        A |= (z[:,rr+dr,cc+dc].astype(np.uint16)<<k)
    A=A.reshape(len(z),49).astype(np.int32)

    has=active.any(axis=1)
    first=np.argmax(active,axis=1)
    gr=(first//7).astype(np.int16)
    gc=(first%7).astype(np.int16)

    dr=GRID_R[None,:]-gr[:,None]
    dc=GRID_C[None,:]-gc[:,None]

    # Only active cells are meaningful. For nonempty shapes the lexicographic
    # anchor guarantees active dr in 0..6, dc in -6..6.
    if np.any(active & ((dr<0)|(dr>6)|(dc<-6)|(dc>6))):
        raise AssertionError("displacement domain violation")

    D=(13*dr+(dc+6)).astype(np.int32)
    J=D*256+A

    Aout=np.where(active,A,-1).astype(np.int32)
    Dout=np.where(active,D,-1).astype(np.int32)
    Jout=np.where(active,J,-1).astype(np.int32)

    # Empty bit-plane emits no observations.
    assert np.all(Aout[~has]==-1)
    return Aout,Dout,Jout

def integration_gate(a):
    checked=0
    for bit in range(8):
        A,D,J=observations(a,bit)
        valid=J>=0
        assert np.all((A[valid]>=0)&(A[valid]<256))
        assert np.all((D[valid]>=0)&(D[valid]<91))
        assert np.array_equal(J[valid],D[valid]*256+A[valid])
        checked+=int(valid.sum())
    return checked

def fit_table(A,y,states):
    cnt=np.zeros((states,K),dtype=np.int32)
    for c in range(K):
        vals=A[y==c]
        vals=vals[vals>=0].astype(np.int64,copy=False)
        cnt[:,c]=np.bincount(vals,minlength=states)
    total=cnt.sum(axis=1,keepdims=True)
    E=K*cnt-total
    return E.astype(np.int32)

def add_score(tab,A,S):
    CH=500
    for st in range(0,len(A),CH):
        en=min(len(A),st+CH)
        x=A[st:en]
        valid=x>=0
        safe=np.where(valid,x,0)
        ev=tab[safe]                 # [B,49,10]
        ev*=valid[...,None]
        S[st:en]+=ev.sum(axis=1,dtype=np.int64)

def accuracy(S,y):
    return float(np.mean(S.argmax(axis=1)==y))

def select_weights(SJ,SA,SD,y):
    best=None
    for wj in (0,1,2,4):
        for wa in (0,1,2,4):
            for wd in (0,1,2,4):
                if wj==wa==wd==0: continue
                a=accuracy(wj*SJ+wa*SA+wd*SD,y)
                key=(a,-(wj+wa+wd),-wj,-wa,-wd)
                if best is None or key>best[0]:
                    best=(key,(wj,wa,wd),a)
    return best[1],best[2]

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=split_40_5_5(train_y)
train_fields=fields(train_x);test_fields=fields(test_x)

# Pre-label integration checks.
integration_observations=0
for name in CHANNEL_NAMES:
    integration_observations += integration_gate(train_fields[name][:256])

yfit=train_y[fit_idx];yval=train_y[val_idx];yshadow=train_y[shadow_idx]

VJ=np.zeros((5000,K),dtype=np.int64); VA=np.zeros_like(VJ); VD=np.zeros_like(VJ)
SJ=np.zeros((5000,K),dtype=np.int64); SA=np.zeros_like(SJ); SD=np.zeros_like(SJ)
TJ=np.zeros((10000,K),dtype=np.int64);TA=np.zeros_like(TJ);TD=np.zeros_like(TJ)

model_bytes={"joint":0,"A":0,"D":0}
progress={}

for ci,name in enumerate(CHANNEL_NAMES):
    tr=train_fields[name];te=test_fields[name]
    for bit in range(8):
        A,D,J=observations(tr,bit)
        tA,tD,tJ=observations(te,bit)

        tabJ=fit_table(J[fit_idx],yfit,J_STATES)
        tabA=fit_table(A[fit_idx],yfit,A_STATES)
        tabD=fit_table(D[fit_idx],yfit,D_STATES)

        model_bytes["joint"]+=tabJ.nbytes
        model_bytes["A"]+=tabA.nbytes
        model_bytes["D"]+=tabD.nbytes

        add_score(tabJ,J[val_idx],VJ);add_score(tabA,A[val_idx],VA);add_score(tabD,D[val_idx],VD)
        add_score(tabJ,J[shadow_idx],SJ);add_score(tabA,A[shadow_idx],SA);add_score(tabD,D[shadow_idx],SD)
        add_score(tabJ,tJ,TJ);add_score(tabA,tA,TA);add_score(tabD,tD,TD)

    w,va=select_weights(VJ,VA,VD,yval)
    progress[name]={
        "channels_used":ci+1,
        "selected_joint_A_D_weights":list(w),
        "validation_accuracy":va,
        "joint_only":accuracy(VJ,yval),
        "A_only":accuracy(VA,yval),
        "D_only":accuracy(VD,yval),
    }
    print(name,progress[name],flush=True)

weights,val_acc=select_weights(VJ,VA,VD,yval)
wj,wa,wd=weights

shadow_scores=wj*SJ+wa*SA+wd*SD
test_scores=wj*TJ+wa*TA+wd*TD

shadow_acc=accuracy(shadow_scores,yshadow)
test_acc=accuracy(test_scores,test_y)

pred=test_scores.argmax(axis=1)
conf=np.zeros((K,K),dtype=np.int64)
for y,p in zip(test_y,pred):
    conf[int(y),int(p)]+=1

result={
    "model":"MPRC-Vision-CIFAR-active-anchor-Arm8",
    "discipline":{
        "gates":[
            "directional_arm8_w2_survival_gate.py",
            "global_local_displacement_survival_gate.py",
            "active_anchor_arm8_serialization_gate.py"
        ],
        "workflow_runs_all_gates_before_training":True,
        "prelabel_integration_observations_checked":integration_observations,
    },
    "dataset":{
        "fit":40000,"validation":5000,"shadow_holdout":5000,"official_test":10000,
        "official_test_status":"exploratory; previously exposed",
        "archive_md5":MD5,
    },
    "representation":{
        "active_only":True,
        "global_anchor":"lexicographic minimum active observer center",
        "D":"translation-normalized active-point displacement, host state 0..90",
        "A":"gated two-depth Arm8 Z256 byte",
        "joint":"host lookup (D,A), 23296 states; not ring arithmetic",
        "observer_grid":"experimental 7x7 stride-4 CIFAR32 sampling",
        "channels":CHANNEL_NAMES,
        "bit_planes":8,
    },
    "learning":{
        "rule":"E_c(s)=10*N_c(s)-N(s)",
        "integer_only":True,
        "gradient_descent":False,
        "softmax":False,
        "weight_choices":[0,1,2,4],
    },
    "validation":{
        "selected_joint_A_D_weights":list(weights),
        "accuracy":val_acc,
        "joint_only":accuracy(VJ,yval),
        "A_only":accuracy(VA,yval),
        "D_only":accuracy(VD,yval),
        "channel_progression":progress,
    },
    "shadow_holdout":{
        "accuracy":shadow_acc,
        "labels_used_for_selection":False,
        "joint_only":accuracy(SJ,yshadow),
        "A_only":accuracy(SA,yshadow),
        "D_only":accuracy(SD,yshadow),
    },
    "exploratory_test":{
        "accuracy":test_acc,
        "joint_only":accuracy(TJ,test_y),
        "A_only":accuracy(TA,test_y),
        "D_only":accuracy(TD,test_y),
        "confusion_matrix":conf.tolist(),
    },
    "model_table_bytes":model_bytes | {"total":sum(model_bytes.values())},
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"All geometry/serialization facts are pre-gated. CIFAR usefulness, channel transform usefulness, and 7x7 sampling are empirical."
}

out=ROOT/"results"/"mprc_vision_cifar_active_anchor_arm8.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
