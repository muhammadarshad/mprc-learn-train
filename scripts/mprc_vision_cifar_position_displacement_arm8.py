"""
MPRC Vision CIFAR — gated Position + Displacement + Arm8 classifier
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
  P       : absolute observer position (49 metadata states)
  D       : normalized displacement from global anchor (91 metadata states)
  A       : local Arm8 context (256 Z256 states)
  P+A     : absolute-position/context joint host lookup (49*256 states)
  D+A     : displacement/context joint host lookup (91*256 states)

P, D, PA and DA are metadata/table addresses only, not Z256 arithmetic states.
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
P_STATES=49
D_STATES=91
A_STATES=256
PA_STATES=P_STATES*A_STATES
DA_STATES=D_STATES*A_STATES

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
    """Return P,D,A,PA,DA arrays [N,49], inactive entries = -1."""
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

    if np.any(active & ((dr<0)|(dr>6)|(dc<-6)|(dc>6))):
        raise AssertionError("displacement domain violation")

    P=(GRID_R*7+GRID_C)[None,:].repeat(len(z),axis=0).astype(np.int32)
    D=(13*dr+(dc+6)).astype(np.int32)
    PA=P*256+A
    DA=D*256+A

    Pout=np.where(active,P,-1).astype(np.int32)
    Dout=np.where(active,D,-1).astype(np.int32)
    Aout=np.where(active,A,-1).astype(np.int32)
    PAout=np.where(active,PA,-1).astype(np.int32)
    DAout=np.where(active,DA,-1).astype(np.int32)

    assert np.all(Aout[~has]==-1)
    return Pout,Dout,Aout,PAout,DAout

def integration_gate(a):
    """Real-data mapping check before labels are read."""
    checked=0
    for bit in range(8):
        P,D,A,PA,DA=observations(a,bit)
        valid=DA>=0
        assert np.all((P[valid]>=0)&(P[valid]<49))
        assert np.all((D[valid]>=0)&(D[valid]<91))
        assert np.all((A[valid]>=0)&(A[valid]<256))
        assert np.array_equal(PA[valid],P[valid]*256+A[valid])
        assert np.array_equal(DA[valid],D[valid]*256+A[valid])
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

def select_weights(SP,SD,SA,SPA,SDA,y):
    """Validation-only branch selection from a deliberately small integer set."""
    choices=(0,1,2)
    best=None
    for wp in choices:
        for wd in choices:
            for wa in choices:
                for wpa in choices:
                    for wda in choices:
                        if wp==wd==wa==wpa==wda==0:
                            continue
                        S=wp*SP+wd*SD+wa*SA+wpa*SPA+wda*SDA
                        a=accuracy(S,y)
                        key=(a,-(wp+wd+wa+wpa+wda),-wp,-wd,-wa,-wpa,-wda)
                        if best is None or key>best[0]:
                            best=(key,(wp,wd,wa,wpa,wda),a)
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

VP=np.zeros((5000,K),dtype=np.int64); VD=np.zeros_like(VP); VA=np.zeros_like(VP)
VPA=np.zeros_like(VP); VDA=np.zeros_like(VP)
SP=np.zeros((5000,K),dtype=np.int64); SD=np.zeros_like(SP); SA=np.zeros_like(SP)
SPA=np.zeros_like(SP); SDA=np.zeros_like(SP)
TP=np.zeros((10000,K),dtype=np.int64);TD=np.zeros_like(TP);TA=np.zeros_like(TP)
TPA=np.zeros_like(TP);TDA=np.zeros_like(TP)

model_bytes={"P":0,"D":0,"A":0,"PA":0,"DA":0}
progress={}

for ci,name in enumerate(CHANNEL_NAMES):
    tr=train_fields[name];te=test_fields[name]
    for bit in range(8):
        P,D,A,PA,DA=observations(tr,bit)
        tP,tD,tA,tPA,tDA=observations(te,bit)

        tabs={
            "P":fit_table(P[fit_idx],yfit,P_STATES),
            "D":fit_table(D[fit_idx],yfit,D_STATES),
            "A":fit_table(A[fit_idx],yfit,A_STATES),
            "PA":fit_table(PA[fit_idx],yfit,PA_STATES),
            "DA":fit_table(DA[fit_idx],yfit,DA_STATES),
        }
        for k,v in tabs.items():
            model_bytes[k]+=v.nbytes

        add_score(tabs["P"],P[val_idx],VP); add_score(tabs["D"],D[val_idx],VD)
        add_score(tabs["A"],A[val_idx],VA); add_score(tabs["PA"],PA[val_idx],VPA)
        add_score(tabs["DA"],DA[val_idx],VDA)

        add_score(tabs["P"],P[shadow_idx],SP); add_score(tabs["D"],D[shadow_idx],SD)
        add_score(tabs["A"],A[shadow_idx],SA); add_score(tabs["PA"],PA[shadow_idx],SPA)
        add_score(tabs["DA"],DA[shadow_idx],SDA)

        add_score(tabs["P"],tP,TP); add_score(tabs["D"],tD,TD)
        add_score(tabs["A"],tA,TA); add_score(tabs["PA"],tPA,TPA)
        add_score(tabs["DA"],tDA,TDA)

    w,va=select_weights(VP,VD,VA,VPA,VDA,yval)
    progress[name]={
        "channels_used":ci+1,
        "selected_P_D_A_PA_DA_weights":list(w),
        "validation_accuracy":va,
        "P_only":accuracy(VP,yval),
        "D_only":accuracy(VD,yval),
        "A_only":accuracy(VA,yval),
        "PA_only":accuracy(VPA,yval),
        "DA_only":accuracy(VDA,yval),
    }
    print(name,progress[name],flush=True)

weights,val_acc=select_weights(VP,VD,VA,VPA,VDA,yval)
wp,wd,wa,wpa,wda=weights

shadow_scores=wp*SP+wd*SD+wa*SA+wpa*SPA+wda*SDA
test_scores=wp*TP+wd*TD+wa*TA+wpa*TPA+wda*TDA
shadow_acc=accuracy(shadow_scores,yshadow)
test_acc=accuracy(test_scores,test_y)

pred=test_scores.argmax(axis=1)
conf=np.zeros((K,K),dtype=np.int64)
for y,p in zip(test_y,pred):
    conf[int(y),int(p)]+=1

result={
    "model":"MPRC-Vision-CIFAR-position-displacement-Arm8",
    "discipline":{
        "required_gate":"position_displacement_arm8_survival_gate.py",
        "workflow_runs_gate_before_training":True,
        "prelabel_integration_observations_checked":integration_observations,
    },
    "dataset":{
        "fit":40000,"validation":5000,"shadow_holdout":5000,"official_test":10000,
        "official_test_status":"exploratory; previously exposed",
        "archive_md5":MD5,
    },
    "representation":{
        "P":"absolute observer position metadata 0..48",
        "D":"displacement from lexicographic active-shape anchor metadata 0..90",
        "A":"two-depth Arm8 Z256 byte",
        "PA":"host lookup (P,A), 12544 states",
        "DA":"host lookup (D,A), 23296 states",
        "identity":"P = G + D",
        "observer_grid":"experimental 7x7 stride-4 CIFAR32 sampling",
        "channels":CHANNEL_NAMES,
        "bit_planes":8,
    },
    "learning":{
        "rule":"E_c(s)=10*N_c(s)-N(s)",
        "integer_only":True,
        "gradient_descent":False,
        "softmax":False,
        "branch_weight_choices":[0,1,2],
    },
    "validation":{
        "selected_P_D_A_PA_DA_weights":list(weights),
        "accuracy":val_acc,
        "P_only":accuracy(VP,yval),
        "D_only":accuracy(VD,yval),
        "A_only":accuracy(VA,yval),
        "PA_only":accuracy(VPA,yval),
        "DA_only":accuracy(VDA,yval),
        "channel_progression":progress,
    },
    "shadow_holdout":{
        "accuracy":shadow_acc,
        "labels_used_for_selection":False,
        "P_only":accuracy(SP,yshadow),
        "D_only":accuracy(SD,yshadow),
        "A_only":accuracy(SA,yshadow),
        "PA_only":accuracy(SPA,yshadow),
        "DA_only":accuracy(SDA,yshadow),
    },
    "exploratory_test":{
        "accuracy":test_acc,
        "P_only":accuracy(TP,test_y),
        "D_only":accuracy(TD,test_y),
        "A_only":accuracy(TA,test_y),
        "PA_only":accuracy(TPA,test_y),
        "DA_only":accuracy(TDA,test_y),
        "confusion_matrix":conf.tolist(),
    },
    "model_table_bytes":model_bytes | {"total":sum(model_bytes.values())},
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"Position/displacement/context serialization is pre-gated. CIFAR usefulness and 7x7 sampling remain empirical."
}

out=ROOT/"results"/"mprc_vision_cifar_position_displacement_arm8.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
