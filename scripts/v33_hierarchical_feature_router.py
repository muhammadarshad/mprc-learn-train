"""v33 — Label-free hierarchical feature router audit.

Purpose
-------
v32 showed:
  per-query ANY-channel oracle top1:
    32x32 = 0.76
    16x16 = 0.81
     8x8  = 0.85
     4x4  = 0.85
     2x2  = 0.83

So hierarchy creates useful feature specialists, but equal channel aggregation
destroys that gain.

v33 asks the missing question:
  can we choose the useful feature channel WITHOUT labels?

No architecture/training change. No class labels enter routing.

For each query, channel k, and scale s:
  E1 = minimum ring MEASURE to memory
  E2 = second-lowest ring MEASURE
  margin = E2-E1

Candidate routers, fixed before seeing labels:
R0 finest-margin:
    choose channel with largest normalized top1/top2 margin at 32.
R1 8-scale-margin:
    choose channel with largest normalized margin at 8.
R2 4-scale-margin:
    choose channel with largest normalized margin at 4.
R3 persistence:
    prefer channels whose nearest MEMORY SAMPLE identity agrees at 8 and 4,
    then largest normalized combined margin.
R4 three-scale persistence:
    prefer count of identical nearest memory sample across 16,8,4,
    then largest normalized combined margin.

Normalization is exact integer cross-comparison by number of sites; no floats
are used for routing. Labels are revealed only after selected memory/sample.

This is a diagnostic routing audit, not a frozen SELECT law.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260929
NCLASS=10; MEM_PER_CLASS=100; QUERY_PER_CLASS=20
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]

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
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y
def load():
    d=ensure(); xs=[];ys=[]
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
    Gx=np.clip(128+gx//2,0,255).astype(np.uint8); Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
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
    dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2]; dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
    Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
    return np.stack([R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl],axis=1)

def cdist4(vals):
    v=vals.astype(np.int16);a=v[..., :,None];b=v[...,None,:]
    d=np.abs(a-b);d=np.minimum(d,256-d)
    return d.sum(axis=-1,dtype=np.int16)
def contract2(x):
    N,C,H,W=x.shape
    cells=x.reshape(N,C,H//2,2,W//2,2).transpose(0,1,2,4,3,5).reshape(N,C,H//2,W//2,4)
    idx=cdist4(cells).argmin(axis=-1)
    return np.take_along_axis(cells,idx[...,None],axis=-1)[...,0].astype(np.uint8)
def hierarchy(x):
    out={x.shape[-1]:x};z=x
    while z.shape[-1]>2:
        z=contract2(z);out[z.shape[-1]]=z
    return out
def energies(q,X):
    a=q.astype(np.int16)[None];b=X.astype(np.int16)
    d=np.abs(a-b);d=np.minimum(d,256-d)
    return d.reshape(len(X),-1).sum(1,dtype=np.int64)

def top2(E):
    # deterministic memory-id tie break
    ids=np.arange(len(E),dtype=np.int64)
    o=np.lexsort((ids,E))
    return int(o[0]),int(E[o[0]]),int(E[o[1]])

def better_margin(m1,sites1,m2,sites2):
    # true iff m1/sites1 > m2/sites2 without floats
    return int(m1)*int(sites2) > int(m2)*int(sites1)

tx,ty,vx,vy=load();rng=np.random.default_rng(SEED)
mem=[]
for c in range(NCLASS):
    ids=np.flatnonzero(ty==c);ids=ids[rng.permutation(len(ids))]
    mem.extend(ids[:MEM_PER_CLASS])
mem=np.asarray(mem,dtype=np.int64); ML=ty[mem]
qids=[]
for c in range(NCLASS):
    ids=np.flatnonzero(vy==c);qids.extend(ids[:QUERY_PER_CLASS])
qids=np.asarray(qids,dtype=np.int64); QL=vy[qids]

MH=hierarchy(fields(tx[mem])); QH=hierarchy(fields(vx[qids]))
SCALES=(32,16,8,4)
sites={s:s*s for s in SCALES}

correct={k:0 for k in ("R0","R1","R2","R3","R4")}
oracle=0
chosen_hist={k:{n:0 for n in CHANNEL_NAMES} for k in correct}
rows=[]

for qi in range(len(qids)):
    data={}
    any_correct=False
    for ch in range(16):
        data[ch]={}
        for s in SCALES:
            E=energies(QH[s][qi,ch],MH[s][:,ch])
            idx,e1,e2=top2(E)
            data[ch][s]={"idx":idx,"e1":e1,"e2":e2,"margin":e2-e1}
            any_correct |= bool(ML[idx]==QL[qi])

    oracle += int(any_correct)

    # R0/R1/R2 choose largest normalized margin at fixed scale
    picks={}
    for rn,s in (("R0",32),("R1",8),("R2",4)):
        best=0
        for ch in range(1,16):
            if better_margin(data[ch][s]["margin"],sites[s],data[best][s]["margin"],sites[s]):
                best=ch
        picks[rn]=(best,data[best][s]["idx"])

    # R3: 8<->4 same nearest sample is primary.
    best=None
    for ch in range(16):
        persist=int(data[ch][8]["idx"]==data[ch][4]["idx"])
        margin_num=data[ch][8]["margin"]*sites[4]+data[ch][4]["margin"]*sites[8]
        # common denominator sites8*sites4
        key=(persist,margin_num,-ch)
        if best is None or key>best[0]:best=(key,ch)
    ch=best[1]
    # if persistent, same idx; otherwise choose scale with stronger normalized margin
    if data[ch][8]["idx"]==data[ch][4]["idx"]:
        idx=data[ch][8]["idx"]
    elif better_margin(data[ch][8]["margin"],sites[8],data[ch][4]["margin"],sites[4]):
        idx=data[ch][8]["idx"]
    else: idx=data[ch][4]["idx"]
    picks["R3"]=(ch,idx)

    # R4: count agreement among 16,8,4 nearest sample identities.
    best=None
    for ch in range(16):
        ids=[data[ch][s]["idx"] for s in (16,8,4)]
        persistence=max(ids.count(z) for z in set(ids))
        # exact common-denominator margin score
        mn=(data[ch][16]["margin"]*sites[8]*sites[4]
            +data[ch][8]["margin"]*sites[16]*sites[4]
            +data[ch][4]["margin"]*sites[16]*sites[8])
        key=(persistence,mn,-ch)
        if best is None or key>best[0]:best=(key,ch)
    ch=best[1]
    ids=[data[ch][s]["idx"] for s in (16,8,4)]
    # majority identity if present; otherwise scale with strongest normalized margin
    vals,counts=np.unique(ids,return_counts=True)
    if int(counts.max())>=2:
        idx=int(vals[np.argmax(counts)])
    else:
        ss=16
        for s in (8,4):
            if better_margin(data[ch][s]["margin"],sites[s],data[ch][ss]["margin"],sites[ss]):ss=s
        idx=data[ch][ss]["idx"]
    picks["R4"]=(ch,idx)

    rr={"query":int(qids[qi]),"truth":NAMES[int(QL[qi])],"picks":{}}
    for rn,(ch,idx) in picks.items():
        ok=bool(ML[idx]==QL[qi]);correct[rn]+=int(ok);chosen_hist[rn][CHANNEL_NAMES[ch]]+=1
        rr["picks"][rn]={"channel":CHANNEL_NAMES[ch],"memory_index":int(idx),
                         "predicted":NAMES[int(ML[idx])],"correct":ok}
    rows.append(rr)

total=len(qids)
report={
  "experiment":"v33 label-free hierarchical feature router",
  "queries":total,
  "memory_images":len(mem),
  "oracle_any_channel_any_tested_scale_rate":oracle/total,
  "routers":{
    rn:{"accuracy":correct[rn]/total,"correct":correct[rn],"channel_histogram":chosen_hist[rn]}
    for rn in correct
  },
  "rows":rows,
  "claim_boundary":"Diagnostic only. Routers are fixed label-free confidence/persistence rules. No router is promoted unless it materially closes the v32 oracle gap and survives a separate holdout."
}
out=ROOT/"results"/"v33_hierarchical_feature_router.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(report,indent=2))
print(json.dumps({"oracle":report["oracle_any_channel_any_tested_scale_rate"],"routers":report["routers"]},indent=2))
