"""
MPRC Vision CIFAR — gated directional Arm8 W2/B1 classifier
============================================================

PRECONDITION:
  scripts/directional_arm8_w2_survival_gate.py must PASS before training.

For each channel bit-plane and each local anchor:

  AU = U1 + 2 U2
  AD = D1 + 2 D2
  AF = F1 + 2 F2
  AB = B1 + 2 B2

  A = AU + 4 AD + 16 AF + 64 AB  in Z256
  W2 = (C,A)                      in Z256^2
  B1 = C-A mod256                 in Z256

The W2 host lookup index is (C<<8)|A in 0..511.
IMPORTANT: this is ONLY a table address.  It is never used as ring arithmetic.

Learning uses integer observation evidence:
  E_c(f,s) = 10*N_c(f,s) - N(f,s)

No gradient descent, no softmax, no float probability, no GPU.
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
        "H1":H1,"H2":H2,"M4":M4,"L8":L8,
        "Contrast":Contrast,"Curl":Curl,
    }

def local_wb(a,bit):
    """Return W lookup [N,49] in 0..511 and B byte [N,49] in 0..255."""
    z=((a>>bit)&1).astype(np.uint8)
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    C=z[:,rr,cc].astype(np.uint16)

    A=np.zeros_like(C,dtype=np.uint16)
    for k,(dr,dc) in enumerate(ARMS):
        A |= (z[:,rr+dr,cc+dc].astype(np.uint16)<<k)

    assert int(A.max())<=255
    W=(C<<8)|A               # host lookup address only
    B=((C-A)&0xFF).astype(np.uint8)

    assert int(W.max())<=511
    return W.reshape(len(z),49),B.reshape(len(z),49)

def integration_gate(a):
    """Real-data mapping check before labels are read."""
    for bit in range(8):
        W,B=local_wb(a,bit)
        C=(W>>8).astype(np.uint16)
        A=(W&0xFF).astype(np.uint16)
        assert np.all(C<=1)
        assert np.array_equal(((C-A)&0xFF).astype(np.uint8),B)
        # (C,B) reconstructs A exactly.
        A2=((C-B.astype(np.uint16))&0xFF)
        assert np.array_equal(A2,A)
    return int(len(a)*49*8)

def fit_evidence(A,y,states):
    F=A.shape[1]
    cnt=np.zeros((F,K,states),dtype=np.int32)
    off=(states*np.arange(F,dtype=np.int64))[None,:]
    for c in range(K):
        R=A[y==c].astype(np.int64,copy=False)
        bc=np.bincount((R+off).ravel(),minlength=F*states).reshape(F,states)
        cnt[:,c,:]=bc
    total=cnt.sum(axis=1,keepdims=True)
    E=K*cnt-total
    return E.transpose(0,2,1).astype(np.int32)

def add_scores(tab,A,S):
    F=A.shape[1]
    fi=np.arange(F,dtype=np.int64)[None,:]
    CH=500
    for st in range(0,len(A),CH):
        en=min(len(A),st+CH)
        S[st:en]+=tab[fi,A[st:en]].sum(axis=1,dtype=np.int64)

def acc(S,y):
    return float(np.mean(S.argmax(axis=1)==y))

def choose_weights(W,B,y):
    best=None
    for ww in (0,1,2,4):
        for wb in (0,1,2,4):
            if ww==0 and wb==0:
                continue
            a=acc(ww*W+wb*B,y)
            key=(a,-(ww+wb),-ww,-wb)
            if best is None or key>best[0]:
                best=(key,(ww,wb),a)
    return best[1],best[2]

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=split_40_5_5(train_y)

train_fields=fields(train_x)
test_fields=fields(test_x)

# Integration gate before labels are consumed.
integration_states=0
for name in CHANNEL_NAMES:
    integration_states+=integration_gate(train_fields[name][:256])

yfit=train_y[fit_idx]
yval=train_y[val_idx]
yshadow=train_y[shadow_idx]

VW=np.zeros((len(val_idx),K),dtype=np.int64)
VB=np.zeros_like(VW)
SW=np.zeros((len(shadow_idx),K),dtype=np.int64)
SB=np.zeros_like(SW)
TW=np.zeros((len(test_y),K),dtype=np.int64)
TB=np.zeros_like(TW)

model_bytes_W=0
model_bytes_B=0
progress={}

for ci,name in enumerate(CHANNEL_NAMES):
    tr=train_fields[name]
    te=test_fields[name]

    for bit in range(8):
        W,B=local_wb(tr,bit)
        tW,tB=local_wb(te,bit)

        tabW=fit_evidence(W[fit_idx],yfit,512)
        tabB=fit_evidence(B[fit_idx],yfit,256)
        model_bytes_W+=tabW.nbytes
        model_bytes_B+=tabB.nbytes

        add_scores(tabW,W[val_idx],VW)
        add_scores(tabW,W[shadow_idx],SW)
        add_scores(tabW,tW,TW)

        add_scores(tabB,B[val_idx],VB)
        add_scores(tabB,B[shadow_idx],SB)
        add_scores(tabB,tB,TB)

    weights,va=choose_weights(VW,VB,yval)
    ww,wb=weights
    progress[name]={
        "channels_used":ci+1,
        "selected_W_B_weights":[ww,wb],
        "validation_accuracy":va,
        "W_only":acc(VW,yval),
        "B_only":acc(VB,yval),
    }
    print(name,progress[name],flush=True)

weights,val_acc=choose_weights(VW,VB,yval)
ww,wb=weights
shadow_scores=ww*SW+wb*SB
test_scores=ww*TW+wb*TB
shadow_acc=acc(shadow_scores,yshadow)
test_acc=acc(test_scores,test_y)

pred=test_scores.argmax(axis=1)
conf=np.zeros((K,K),dtype=np.int64)
for y,p in zip(test_y,pred):
    conf[int(y),int(p)]+=1

result={
    "model":"MPRC-Vision-CIFAR-directional-Arm8-W2-B1",
    "discipline":{
        "required_gate":"directional_arm8_w2_survival_gate.py",
        "workflow_runs_gate_before_training":True,
        "real_data_prelabel_integration_states_checked":integration_states,
        "Z512_ring_state_used":False,
    },
    "dataset":{
        "fit":40000,
        "validation":5000,
        "shadow_holdout":5000,
        "official_test":10000,
        "official_test_status":"exploratory; previously exposed",
        "archive_md5":MD5,
    },
    "geometry":{
        "context":"C + (U1,U2)+(D1,D2)+(F1,F2)+(B1,B2)",
        "Arm8_byte":"U1+2U2+4D1+8D2+16F1+32F2+64B1+128B2",
        "W2":"(C,Arm8) in Z256^2",
        "B1":"C-Arm8 mod256",
        "W_lookup_index":"(C<<8)|Arm8, host address only",
        "anchors":49,
        "channels":CHANNEL_NAMES,
        "bit_planes":8,
    },
    "learning":{
        "rule":"E_c(f,s)=10*N_c(f,s)-N(f,s)",
        "table_dtype":"int32 exact",
        "gradient_descent":False,
        "softmax":False,
        "branch_weight_choices":[0,1,2,4],
    },
    "validation":{
        "selected_W_B_weights":[ww,wb],
        "accuracy":val_acc,
        "W_only_accuracy":acc(VW,yval),
        "B_only_accuracy":acc(VB,yval),
        "channel_progression":progress,
    },
    "shadow_holdout":{
        "accuracy":shadow_acc,
        "W_only_accuracy":acc(SW,yshadow),
        "B_only_accuracy":acc(SB,yshadow),
        "labels_used_for_selection":False,
    },
    "exploratory_test":{
        "accuracy":test_acc,
        "W_only_accuracy":acc(TW,test_y),
        "B_only_accuracy":acc(TB,test_y),
        "confusion_matrix":conf.tolist(),
    },
    "model_table_bytes":{
        "W2":model_bytes_W,
        "B1":model_bytes_B,
        "total":model_bytes_W+model_bytes_B,
    },
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"Arm8/W2/B1 finite encoding facts are gated. Classification accuracy and CIFAR32 49-anchor sampling remain empirical."
}

out=ROOT/"results"/"mprc_vision_cifar_directional_arm8_w2_b1.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
