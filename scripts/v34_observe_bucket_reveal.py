"""v34 — Observe -> Bucket -> Reveal semantic audit.

PHASE 1: UNSUPERVISED OBSERVATION
---------------------------------
CIFAR labels are not consulted while extracting features or forming buckets.

Pipeline:
    image
      -> 16 deterministic feature channels
      -> hierarchical contraction 32->16->8->4
      -> per-(channel,scale) ring-distance nearest occurrence
      -> mutual-nearest-neighbour (MNN) recurrence graph
      -> connected components = structural buckets

Why MNN:
- no class labels
- no learned weights
- no top-K
- no arbitrary distance threshold
- edges only when two observations select each other under an existing
  feature/scale view.

We keep every feature/scale relation separate. An edge may therefore carry
multiple votes (different channels/scales independently produce the same MNN
relation). For the first audit, ONE MNN relation is sufficient to connect a
component; edge vote count is only reported, not thresholded.

PHASE 2: REVEAL
---------------
Only after bucket IDs are frozen do we reveal labels and compute:
- bucket purity
- class coverage
- weighted purity
- bucket label entropy
- how many classes fragment into multiple structural buckets
- how many buckets mix classes

This does NOT train a classifier and does NOT use labels to alter buckets.
It tests the user's hypothesis:
    observe -> organize -> human/semantic binding later.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request,math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260929
NCLASS=10
SAMPLE_PER_CLASS=150
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]
SCALES=(16,8,4)

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
def load_train():
    d=ensure();xs=[];ys=[]
    for i in range(1,6):
        x,y=lb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
    return np.concatenate(xs),np.concatenate(ys)

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
    while z.shape[-1]>4:
        z=contract2(z);out[z.shape[-1]]=z
    return out

def pairwise_nearest(X,chunk=64):
    # X [N,H,W] uint8. Exact ring-L1. Self excluded. Deterministic tie -> smallest id.
    N=len(X);flat=X.reshape(N,-1).astype(np.int16)
    nearest=np.empty(N,dtype=np.int32);bestE=np.empty(N,dtype=np.int64)
    for st in range(0,N,chunk):
        q=flat[st:st+chunk]
        # [Q,N,F]
        d=np.abs(q[:,None,:]-flat[None,:,:]);d=np.minimum(d,256-d)
        E=d.sum(axis=2,dtype=np.int64)
        rows=np.arange(len(q));cols=np.arange(st,min(N,st+chunk))
        E[rows,cols]=np.iinfo(np.int64).max
        nearest[st:st+len(q)]=E.argmin(axis=1)
        bestE[st:st+len(q)]=E[rows,nearest[st:st+len(q)]]
    return nearest,bestE

class DSU:
    def __init__(self,n):
        self.p=list(range(n));self.sz=[1]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a,b):
        a=self.find(a);b=self.find(b)
        if a==b:return
        if self.sz[a]<self.sz[b]:a,b=b,a
        self.p[b]=a;self.sz[a]+=self.sz[b]

x,y=load_train()
# Balanced sample is chosen using labels ONLY to avoid class-imbalance in the audit population.
# Labels are discarded before feature extraction/bucket formation.
rng=np.random.default_rng(SEED)
ids=[]
for c in range(NCLASS):
    z=np.flatnonzero(y==c);z=z[rng.permutation(len(z))]
    ids.extend(z[:SAMPLE_PER_CLASS])
ids=np.asarray(sorted(ids),dtype=np.int64)
X=x[ids]
hidden_labels=y[ids].copy()

F=fields(X)
H=hierarchy(F)
N=len(X)
dsu=DSU(N)
edge_votes={}
view_stats=[]

# ---------- PHASE 1: NO LABEL ACCESS BELOW THIS LINE ----------
for s in SCALES:
    for ch,name in enumerate(CHANNEL_NAMES):
        nn,e=pairwise_nearest(H[s][:,ch])
        mutual=0
        for i,j0 in enumerate(nn):
            j=int(j0)
            if int(nn[j])==i and i<j:
                mutual+=1
                dsu.union(i,j)
                key=(i,j)
                edge_votes[key]=edge_votes.get(key,0)+1
        view_stats.append({"scale":s,"channel":name,"mutual_pairs":mutual,
                           "mean_nearest_energy":float(e.mean())})

roots=[dsu.find(i) for i in range(N)]
root_to_bucket={}
bucket_ids=np.empty(N,dtype=np.int32)
for i,r in enumerate(roots):
    if r not in root_to_bucket:root_to_bucket[r]=len(root_to_bucket)
    bucket_ids[i]=root_to_bucket[r]

# freeze observable bucket artifact BEFORE semantic reveal
bucket_members={}
for i,b0 in enumerate(bucket_ids):
    bucket_members.setdefault(int(b0),[]).append(int(i))
phase1={
    "n_images":N,
    "n_buckets":len(bucket_members),
    "bucket_sizes":sorted([len(v) for v in bucket_members.values()],reverse=True),
    "n_mnn_edges":len(edge_votes),
    "edge_vote_histogram":{},
    "views":view_stats,
    "bucket_ids":bucket_ids.tolist(),
}
for v in edge_votes.values():
    phase1["edge_vote_histogram"][str(v)]=phase1["edge_vote_histogram"].get(str(v),0)+1

# ---------- PHASE 2: LABELS REVEALED ONLY NOW ----------
labels=hidden_labels
bucket_report=[]
weighted_correct=0
ent_weight=0.0
mixed=0
pure=0
nontrivial=0
class_buckets={c:set() for c in range(NCLASS)}
for b,members in sorted(bucket_members.items()):
    yy=labels[np.asarray(members)]
    cnt=np.bincount(yy,minlength=NCLASS)
    maj=int(cnt.argmax());maj_n=int(cnt[maj]);size=len(members)
    purity=maj_n/size
    probs=cnt[cnt>0]/size
    entropy=float(-(probs*np.log2(probs)).sum()) if size>1 else 0.0
    weighted_correct+=maj_n;ent_weight+=size*entropy
    if size>1: nontrivial+=1
    if np.count_nonzero(cnt)>1:mixed+=1
    else:pure+=1
    for c in np.flatnonzero(cnt):class_buckets[int(c)].add(int(b))
    bucket_report.append({
        "bucket":int(b),"size":size,"majority_class":NAMES[maj],"purity":purity,
        "entropy_bits":entropy,
        "class_counts":{NAMES[c]:int(cnt[c]) for c in range(NCLASS) if cnt[c]}
    })

# metrics on non-singleton buckets separately, because singleton purity is trivial
non_single=[r for r in bucket_report if r["size"]>1]
ns_total=sum(r["size"] for r in non_single)
ns_correct=sum(round(r["purity"]*r["size"]) for r in non_single)

result={
 "experiment":"v34 observe-bucket-reveal semantic audit",
 "phase1_unsupervised":phase1,
 "phase2_reveal":{
   "weighted_bucket_purity_all":weighted_correct/N,
   "mean_bucket_entropy_bits_weighted":ent_weight/N,
   "pure_buckets":pure,"mixed_buckets":mixed,"nontrivial_buckets":nontrivial,
   "non_singleton_images":ns_total,
   "weighted_purity_non_singleton":(ns_correct/ns_total if ns_total else None),
   "class_fragmentation":{NAMES[c]:len(class_buckets[c]) for c in range(NCLASS)},
   "largest_buckets":sorted(bucket_report,key=lambda r:(-r["size"],r["bucket"]))[:50]
 },
 "claim_boundary":"Buckets are formed without semantic labels. Balanced sampling uses labels only before the blind phase to make the audit population class-balanced. This is not a trained classifier; post-hoc purity measures whether observation-only recurrence aligns with semantic classes."
}
out=ROOT/"results"/"v34_observe_bucket_reveal.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(result,indent=2))
print(json.dumps({
 "phase1":{"n_images":N,"n_buckets":phase1["n_buckets"],"largest_sizes":phase1["bucket_sizes"][:20],
           "n_mnn_edges":phase1["n_mnn_edges"],"edge_vote_histogram":phase1["edge_vote_histogram"]},
 "phase2":{k:v for k,v in result["phase2_reveal"].items() if k!="largest_buckets"}
},indent=2))
