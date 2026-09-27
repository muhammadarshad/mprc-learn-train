"""CIFAR-10 corrected Arshad-ViT streaming byte-state audit.

SOURCE hierarchy
----------------
kernel : 3x3 = 9 unique source pixels
patch  : 16x7 = 112 unique source pixels
local  : eight disjoint 16x7 patches = 32x28 = 896 unique source pixels
global : local + two 16x4 residual strips = full 32x32 = 1024 pixels

STORAGE / COMPUTE
-----------------
storage     : 128x113 bits = 14,464 bits
byte state  : 16x113 Z256 bytes = 1,808 bytes

Patch observations are never called storage. Each native 16x7 RGB patch contains
336 source bytes and is injected into the fixed byte state:
  rows 0..15, cols 0..6   = R
               7..13      = G
              14..20      = B

Patch position P is represented by a generator-7 column transport:
  shift_j = 7*j mod 113.
This is a coordinate transport, not extra image data.

LOCAL streams 8 exact native patches through:
  state <- REACT(BIND(state, transported_patch))
GLOBAL then streams the two remaining 16x4 RGB strips, covering the 128 source
pixels not present in the 32x28 local tile.

KERNEL and PATCH are encoded once and REACTed once.

Evaluation is a label-blind nearest-memory control under exact Z256 circular
MEASURE on the retained 1,808-byte state. Labels are read only after selection.
This is an empirical candidate for the corrected execution path, not a theorem.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

from mprc_structural.byte_attention import bind,react,measure
from mprc_structural.bit_manifold import pack_bytes,unpack_bytes,STORAGE_BYTES,MANIFOLD_BITS

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10";CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz";MD5="c58f30108f718f92721af3b95e74349a"
SEED=20260927
MEMORY=1000
QUERIES=500
GEN=7
W=113

def md5(p):
    h=hashlib.md5()
    with open(p,"rb") as f:
        for z in iter(lambda:f.read(1<<20),b""):h.update(z)
    return h.hexdigest()

def data_dir():
    if not ARCH.exists() or md5(ARCH)!=MD5:urllib.request.urlretrieve(URL,ARCH)
    assert md5(ARCH)==MD5
    d=CACHE/"cifar-10-batches-py"
    if not d.exists():
        with tarfile.open(ARCH,"r:gz") as tf:tf.extractall(CACHE)
    return d

def load_batch(p):
    with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def shift_obs(obs,j):
    return np.roll(obs,(GEN*int(j))%W,axis=1)

def encode_kernel(img):
    out=np.zeros((16,113),dtype=np.uint8)
    p=img[15:18,15:18,:]  # [3,3,3]
    # Preserve all 27 bytes exactly in an explicit 3 x (3*RGB) view.
    out[0:3,0:3]=p[:,:,0]
    out[0:3,3:6]=p[:,:,1]
    out[0:3,6:9]=p[:,:,2]
    return react(out,rounds=1)

def patch_obs(patch16x7):
    p=np.asarray(patch16x7)
    assert p.shape==(16,7,3) and p.dtype==np.uint8
    out=np.zeros((16,113),dtype=np.uint8)
    out[:,0:7]=p[:,:,0]
    out[:,7:14]=p[:,:,1]
    out[:,14:21]=p[:,:,2]
    return out

def residual_obs(strip16x4):
    p=np.asarray(strip16x4)
    assert p.shape==(16,4,3) and p.dtype==np.uint8
    out=np.zeros((16,113),dtype=np.uint8)
    out[:,0:4]=p[:,:,0]
    out[:,4:8]=p[:,:,1]
    out[:,8:12]=p[:,:,2]
    return out

def encode_patch(img):
    # One central native patch.
    o=patch_obs(img[8:24,12:19,:])
    return react(o,rounds=1)

LOCAL_PATCHES=[(br,bc) for br in (0,16) for bc in (2,9,16,23)]
assert len(LOCAL_PATCHES)==8

def encode_local(img):
    state=np.zeros((16,113),dtype=np.uint8)
    for j,(r,c) in enumerate(LOCAL_PATCHES):
        o=shift_obs(patch_obs(img[r:r+16,c:c+7,:]),j)
        state=react(bind(state,o),rounds=1)
    return state

def encode_global(img):
    state=encode_local(img)
    # Remaining source columns: 0,1,30,31 => two 16x4 row strips.
    cols=[0,1,30,31]
    for k,r0 in enumerate((0,16),start=8):
        strip=img[r0:r0+16,cols,:]
        o=shift_obs(residual_obs(strip),k)
        state=react(bind(state,o),rounds=1)
    return state

def ring_dist_batch(mem,q):
    d=np.abs(mem.astype(np.int16)-q.astype(np.int16)[None,:,:])
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

encoders={
  "kernel9":(encode_kernel,9),
  "patch112":(encode_patch,112),
  "local896":(encode_local,896),
  "global1024":(encode_global,1024),
}
metrics={}
preds={}

for name,(fn,pixels) in encoders.items():
    mem=np.stack([fn(train_x[i]) for i in midx],axis=0)
    # Storage roundtrip proves the retained computational state fits exactly.
    for z in mem[:32]:
        assert np.array_equal(unpack_bytes(pack_bytes(z)),z)
    correct=0; pv=[]; margins=[]
    for qi in qidx:
        q=fn(test_x[qi])
        e=ring_dist_batch(mem,q)
        two=np.argpartition(e,1)[:2]
        two=two[np.argsort(e[two],kind="stable")]
        w=int(two[0]);r=int(two[1])
        pred=int(train_y[midx[w]]);truth=int(test_y[qi])
        correct+=int(pred==truth);pv.append(pred);margins.append(int(e[r]-e[w]))
    source_bytes=3*pixels
    metrics[name]={
      "unique_source_pixels":pixels,
      "source_rgb_bytes_observed":source_bytes,
      "retained_state_bytes":STORAGE_BYTES,
      "top1_accuracy":correct/len(qidx),
      "correct":correct,
      "queries":len(qidx),
      "median_measure_margin":float(np.median(np.asarray(margins,dtype=np.int64))),
      "raw_source_bytes_fit_simultaneously":bool(source_bytes<=STORAGE_BYTES),
      "streaming_required":bool(source_bytes>STORAGE_BYTES),
    }
    preds[name]=np.asarray(pv,dtype=np.int16)

truth=test_y[qidx]
transitions={}
names=list(encoders)
for a,b in zip(names[:-1],names[1:]):
    aa=preds[a]==truth;bb=preds[b]==truth
    transitions[f"{a}->{b}"]={
      "wrong_to_correct":int(np.count_nonzero((~aa)&bb)),
      "correct_to_wrong":int(np.count_nonzero(aa&(~bb))),
      "prediction_changed":int(np.count_nonzero(preds[a]!=preds[b])),
    }

R={
 "experiment":"Corrected Arshad-ViT byte-state streaming CIFAR audit",
 "status":"COMPLETE",
 "dataset":{"name":"official CIFAR-10","memory":MEMORY,"queries":QUERIES},
 "storage":{"bits":MANIFOLD_BITS,"bytes":STORAGE_BYTES,"compute_shape":[16,113]},
 "hierarchy":{"kernel":9,"patch":112,"local":896,"global":1024},
 "local_stream":{"native_patch":[16,7],"patches":8,"pixels":896},
 "global_stream":{"residual_pixels":128,"total_pixels":1024},
 "position_transport":"column roll by 7*j mod113",
 "update":"state <- REACT(BIND(state, observation)), one round per streamed observation",
 "metrics":metrics,
 "transitions":transitions,
 "discipline":{
   "source_pixel_counts_distinct_from_computational_support":True,
   "storage_bits_not_used_as_Z256_bytes":True,
   "labels_used_for_encoding_or_selection":False,
   "softmax":False,"gradient_descent":False,"resize":False
 },
 "claim_boundary":"Empirical candidate streaming codec + exact ring nearest-memory readout. Patch placement and one-round-per-observation policy are candidates, not frozen MPRC theorems."
}
out=ROOT/"results"/"cifar_corrected_byte_stream.json";out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
