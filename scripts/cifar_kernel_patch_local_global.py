"""CIFAR-10 bottom-up source-pixel hierarchy audit.

This audit keeps SOURCE INFORMATION distinct from STORAGE:

KERNEL : 3x3 = 9 unique source pixels
PATCH  : 16x7 = 112 unique source pixels
LOCAL  : 8 native 16x7 patches = 32x28 = 896 unique source pixels
GLOBAL : full CIFAR image = 32x32 = 1024 unique source pixels

All counts are spatial source pixels. RGB source bytes = 3*pixels.
The corrected Arshad-ViT storage capacity remains 14,464 bits = 1,808 bytes.

No resizing, no learned weights, no softmax.  For a label-blind control, each
query is matched to the nearest memory image under exact Z256 circular L1 over
the selected source pixels. Labels are read only after the winner is chosen.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np
from mprc_structural.bit_manifold import STORAGE_BYTES,MANIFOLD_BITS

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260927
MEMORY=5000
QUERIES=1000

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for z in iter(lambda:f.read(1<<20),b""): h.update(z)
    return h.hexdigest()

def data_dir():
    if not ARCH.exists() or md5(ARCH)!=MD5: urllib.request.urlretrieve(URL,ARCH)
    assert md5(ARCH)==MD5
    d=CACHE/"cifar-10-batches-py"
    if not d.exists():
        with tarfile.open(ARCH,"r:gz") as tf: tf.extractall(CACHE)
    return d

def load_batch(p):
    with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def ring_l1(mem,q):
    d=np.abs(mem.astype(np.int16)-q.astype(np.int16)[None,...])
    d=np.minimum(d,256-d)
    return d.sum(axis=(1,2),dtype=np.int64)

d=data_dir()
xs=[];ys=[]
for i in range(1,6):
    x,y=load_batch(d/f"data_batch_{i}");xs.append(x);ys.append(y)
train_x=np.concatenate(xs);train_y=np.concatenate(ys)
test_x,test_y=load_batch(d/"test_batch")

rng=np.random.default_rng(SEED)
midx=rng.permutation(len(train_x))[:MEMORY]
qidx=rng.permutation(len(test_x))[:QUERIES]

# KERNEL: one actual 3x3 source neighborhood around image center.
kernel=np.asarray([(r,c) for r in range(15,18) for c in range(15,18)],dtype=np.int16)
assert len(kernel)==9

# PATCH: one exact native 16x7 source rectangle.
patch=np.asarray([(r,c) for r in range(8,24) for c in range(12,19)],dtype=np.int16)
assert len(patch)==112

# LOCAL: eight disjoint native 16x7 patches tile 32x28.
local=np.asarray([(r,c) for r in range(32) for c in range(2,30)],dtype=np.int16)
assert len(local)==896
# prove exact decomposition into 8 non-overlapping 16x7 patches
parts=[]
for br in (0,16):
    for bc in (2,9,16,23):
        p={(r,c) for r in range(br,br+16) for c in range(bc,bc+7)}
        parts.append(p)
assert len(parts)==8
assert len(set().union(*parts))==896
assert sum(map(len,parts))==896

# GLOBAL: every source pixel.
global_=np.asarray([(r,c) for r in range(32) for c in range(32)],dtype=np.int16)
assert len(global_)==1024

stages={"kernel9":kernel,"patch112":patch,"local896":local,"global1024":global_}
metrics={}
predictions={}

for name,coords in stages.items():
    rr=coords[:,0];cc=coords[:,1]
    mem=train_x[midx][:,rr,cc,:]
    correct=0
    preds=[]
    margins=[]
    for qi in qidx:
        q=test_x[qi,rr,cc,:]
        e=ring_l1(mem,q)
        order=np.argpartition(e,1)[:2]
        order=order[np.argsort(e[order],kind="stable")]
        winner=int(order[0]); runner=int(order[1])
        pred=int(train_y[midx[winner]]); truth=int(test_y[qi])
        correct+=int(pred==truth)
        preds.append(pred)
        margins.append(int(e[runner]-e[winner]))
    P=len(coords); B=3*P
    metrics[name]={
        "unique_source_pixels":P,
        "source_rgb_bytes":B,
        "storage_capacity_bytes":STORAGE_BYTES,
        "source_bytes_over_storage_capacity":B/STORAGE_BYTES,
        "fits_in_single_1808B_storage_if_raw_rgb":bool(B<=STORAGE_BYTES),
        "top1_accuracy":correct/len(qidx),
        "correct":correct,
        "queries":len(qidx),
        "median_nearest_margin":float(np.median(np.asarray(margins,dtype=np.int64))),
    }
    predictions[name]=np.asarray(preds,dtype=np.int16)

# How often adding the next information scale corrects vs breaks a prior decision.
transitions={}
order=["kernel9","patch112","local896","global1024"]
truth=test_y[qidx]
for a,b in zip(order[:-1],order[1:]):
    pa=predictions[a];pb=predictions[b]
    aok=pa==truth; bok=pb==truth
    transitions[f"{a}->{b}"]={
        "wrong_to_correct":int(np.count_nonzero((~aok)&bok)),
        "correct_to_wrong":int(np.count_nonzero(aok&(~bok))),
        "prediction_changed":int(np.count_nonzero(pa!=pb)),
    }

R={
 "experiment":"CIFAR-10 kernel->patch->local->global source-information audit",
 "status":"COMPLETE",
 "dataset":{"name":"CIFAR-10","memory_images":MEMORY,"query_images":QUERIES,"shape":[32,32,3]},
 "storage":{"bits":MANIFOLD_BITS,"bytes":STORAGE_BYTES,"note":"capacity only; not source-pixel count"},
 "hierarchy":{
   "kernel":{"shape":[3,3],"pixels":9},
   "patch":{"shape":[16,7],"pixels":112},
   "local":{"native_patches":8,"tile_shape":[32,28],"pixels":896},
   "global":{"shape":[32,32],"pixels":1024}
 },
 "metrics":metrics,
 "transitions":transitions,
 "discipline":{
   "pixels_are_unique_source_coordinates":True,
   "labels_used_for_selection":False,
   "resize":False,
   "softmax":False,
   "training":False,
   "distance":"exact Z256 circular L1"
 },
 "claim_boundary":"Nearest-memory control only. It measures information gained by larger source-pixel scopes; it is not yet the corrected BIND->REACT->MEASURE ViT classifier."
}
out=ROOT/"results"/"cifar_kernel_patch_local_global.json";out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
