"""v32 — Hierarchical spatial contraction audit for MPRC vision.

CANDIDATE ENCODER STAGE. No frozen MPRC equation is changed.

Goal
----
Test the missing hierarchical idea directly:
    fine feature map -> coarser feature map -> ... -> global structural evidence

For CIFAR32 we test the homologous dyadic ladder:
    32 -> 16 -> 8 -> 4 -> 2

The same operator maps the canonical 112 payload as:
    112 -> 56 -> 28 -> 14 -> 7

Important:
- 16 deterministic feature channels are preserved independently.
- No channel summation is used during contraction.
- No interpolation, averaging, softmax, gradient descent, labels, or learned weights.
- The contraction is a precommitted ring-native candidate:
    for each 2x2 cell, choose the EXISTING byte minimizing total d256
    to the other three bytes (circular medoid).
  Deterministic tie-break: smallest flattened child index.

This is not claimed as the final MPRC pooling law. It tests whether a hierarchical
spatial contraction itself improves semantic neighbourhood before ViT/attention.

Outputs
-------
For each scale and channel:
- top1 same-class rate on 100 fixed CIFAR queries against balanced memory
- mean first same-class rank
- same-class count in top10

Also report:
- equal-channel aggregate at each scale (diagnostic only)
- per-query best channel at each scale (oracle diagnostic, NOT architecture)
- whether some scale consistently improves over scale0.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260929
NCLASS=10
MEM_PER_CLASS=100
QUERY_PER_CLASS=10
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def ensure():
    if not ARCH.exists() or md5(ARCH)!=MD5:
        urllib.request.urlretrieve(URL,ARCH)
    d=CACHE/"cifar-10-batches-py"
    if not d.exists():
        with tarfile.open(ARCH,"r:gz") as tf: tf.extractall(CACHE)
    return d

def lb(p):
    with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def load():
    d=ensure();xs=[];ys=[]
    for i in range(1,6):
        x,y=lb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
    tx=np.concatenate(xs);ty=np.concatenate(ys);vx,vy=lb(d/"test_batch")
    return tx,ty,vx,vy

def luma(x):
    r=x[...,0].astype(np.uint16);g=x[...,1].astype(np.uint16);b=x[...,2].astype(np.uint16)
    return ((77*r+150*g+29*b)>>8).astype(np.uint8)

def blur(a,r):
    p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
    I=np.pad(p,((0,0),(1,0),(1,0)),mode="constant").cumsum(1,dtype=np.uint32).cumsum(2,dtype=np.uint32)
    k=2*r+1
    s=I[:,k:,k:]-I[:,:-k,k:]-I[:,k:,:-k]+I[:,:-k,:-k]
    return ((s+k*k//2)//(k*k)).astype(np.uint8)

def fields(x):
    R=x[...,0];G=x[...,1];B=x[...,2];Y=luma(x)
    Gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
    Chroma=(x.max(-1).astype(np.int16)-x.min(-1).astype(np.int16)).astype(np.uint8)
    gx=np.zeros_like(Y,dtype=np.int16);gy=np.zeros_like(Y,dtype=np.int16)
    gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
    gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)
    Gx=np.clip(128+gx//2,0,255).astype(np.uint8)
    Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
    Grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)
    lap=np.zeros_like(Y,dtype=np.int16);c=Y[:,1:-1,1:-1].astype(np.int16)
    lap[:,1:-1,1:-1]=(Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c)
    Lap=np.clip(128+lap//4,0,255).astype(np.uint8)
    b1=blur(Y,1);b2=blur(Y,2);b4=blur(Y,4);b8=blur(Y,8)
    H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
    H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
    M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
    Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
    dgy=np.zeros_like(Y,dtype=np.int16);dgx=np.zeros_like(Y,dtype=np.int16)
    dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2]
    dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
    return np.stack([R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl],axis=1)

def cdist4(vals):
    # vals [...,4] uint8; return total circular distance from each child to all four.
    v=vals.astype(np.int16)
    a=v[..., :, None]
    b=v[..., None, :]
    d=np.abs(a-b)
    d=np.minimum(d,256-d)
    return d.sum(axis=-1,dtype=np.int16)

def contract2(x):
    # x [N,C,H,W], even H,W. Choose existing 2x2 child circular medoid.
    N,C,H,W=x.shape
    assert H%2==0 and W%2==0
    cells=x.reshape(N,C,H//2,2,W//2,2).transpose(0,1,2,4,3,5).reshape(N,C,H//2,W//2,4)
    cost=cdist4(cells)
    idx=cost.argmin(axis=-1)  # deterministic first child tie-break
    out=np.take_along_axis(cells,idx[...,None],axis=-1)[...,0]
    return out.astype(np.uint8)

def hierarchy(x):
    out={str(x.shape[-1]):x}
    z=x
    while z.shape[-1]>2:
        z=contract2(z)
        out[str(z.shape[-1])]=z
    return out

def ring_energy_rows(q,X):
    qa=q.astype(np.int16)[None]
    xa=X.astype(np.int16)
    d=np.abs(xa-qa)
    d=np.minimum(d,256-d)
    return d.reshape(len(X),-1).sum(axis=1,dtype=np.int64)

def rank_metrics(E,labels,truth):
    o=np.argsort(E,kind="stable")
    labs=labels[o]
    same=np.flatnonzero(labs==truth)
    return {
        "top1_same": bool(labs[0]==truth),
        "first_same_rank": int(same[0]+1) if len(same) else None,
        "same_in_top10": int((labs[:10]==truth).sum()),
    }

tx,ty,vx,vy=load()
rng=np.random.default_rng(SEED)
mem=[]
for c in range(NCLASS):
    ids=np.flatnonzero(ty==c); ids=ids[rng.permutation(len(ids))]
    mem.extend(ids[:MEM_PER_CLASS])
mem=np.asarray(mem,dtype=np.int64)
qids=[]
for c in range(NCLASS):
    ids=np.flatnonzero(vy==c)
    qids.extend(ids[:QUERY_PER_CLASS])
qids=np.asarray(qids,dtype=np.int64)

M=fields(tx[mem]); Q=fields(vx[qids])
ML=ty[mem]; QL=vy[qids]
MH=hierarchy(M); QH=hierarchy(Q)
scales=sorted(MH.keys(),key=lambda s:-int(s))

report={"experiment":"v32 hierarchical spatial contraction audit",
        "candidate_operator":"2x2 circular-medoid representative; existing byte only",
        "cifar_scales":scales,
        "canonical_112_analogue":[112,56,28,14,7],
        "channels":CHANNEL_NAMES,
        "scales":{}}

for sc in scales:
    m=MH[sc];q=QH[sc]
    per_ch={}
    equal={"top1_same":0,"first_same_rank_sum":0,"same_in_top10_sum":0}
    oracle={"top1_same":0}
    for k,name in enumerate(CHANNEL_NAMES):
        top=0;rank_sum=0;t10=0
        for i in range(len(q)):
            E=ring_energy_rows(q[i,k],m[:,k])
            z=rank_metrics(E,ML,int(QL[i]))
            top+=int(z["top1_same"]); rank_sum+=int(z["first_same_rank"]); t10+=z["same_in_top10"]
        per_ch[name]={
            "top1_same_rate":top/len(q),
            "mean_first_same_rank":rank_sum/len(q),
            "mean_same_in_top10":t10/len(q),
        }

    for i in range(len(q)):
        # equal-channel aggregate is diagnostic; preserve channels up to this point.
        Eall=np.zeros(len(m),dtype=np.int64)
        channel_correct=[]
        for k in range(len(CHANNEL_NAMES)):
            Ek=ring_energy_rows(q[i,k],m[:,k])
            Eall+=Ek
            channel_correct.append(bool(ML[np.argmin(Ek)]==QL[i]))
        z=rank_metrics(Eall,ML,int(QL[i]))
        equal["top1_same"]+=int(z["top1_same"])
        equal["first_same_rank_sum"]+=int(z["first_same_rank"])
        equal["same_in_top10_sum"]+=z["same_in_top10"]
        oracle["top1_same"]+=int(any(channel_correct))

    report["scales"][sc]={
        "spatial":[int(sc),int(sc)],
        "per_channel":per_ch,
        "equal_channel_aggregate":{
            "top1_same_rate":equal["top1_same"]/len(q),
            "mean_first_same_rank":equal["first_same_rank_sum"]/len(q),
            "mean_same_in_top10":equal["same_in_top10_sum"]/len(q),
        },
        "per_query_any_channel_oracle_top1_rate":oracle["top1_same"]/len(q),
    }
    best=max(per_ch.items(),key=lambda kv:kv[1]["top1_same_rate"])
    print("SCALE",sc,"BEST",best[0],best[1],"EQUAL",report["scales"][sc]["equal_channel_aggregate"],"ORACLE",report["scales"][sc]["per_query_any_channel_oracle_top1_rate"],flush=True)

# Improvement table relative to finest scale.
fine=scales[0]
base=report["scales"][fine]
improve={}
for sc in scales[1:]:
    improve[sc]={
        "equal_top1_delta":report["scales"][sc]["equal_channel_aggregate"]["top1_same_rate"]-base["equal_channel_aggregate"]["top1_same_rate"],
        "oracle_top1_delta":report["scales"][sc]["per_query_any_channel_oracle_top1_rate"]-base["per_query_any_channel_oracle_top1_rate"],
        "channels_improved_top1":[
            n for n in CHANNEL_NAMES
            if report["scales"][sc]["per_channel"][n]["top1_same_rate"]>base["per_channel"][n]["top1_same_rate"]
        ]
    }
report["improvement_vs_finest"]=improve
report["claim_boundary"]="This tests one label-free hierarchical contraction candidate. It does not freeze circular-medoid contraction as the MPRC hierarchy law. A positive result only establishes that spatial contraction can improve reusable feature neighbourhoods."

out=ROOT/"results"/"v32_hierarchical_spatial_contraction.json"
out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(report,indent=2))
print(json.dumps({"scales":{s:{"equal":report["scales"][s]["equal_channel_aggregate"],"oracle":report["scales"][s]["per_query_any_channel_oracle_top1_rate"]} for s in scales},"improvement":improve},indent=2))
