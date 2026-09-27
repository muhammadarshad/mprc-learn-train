"""CIFAR-10 source-information audit with corrected Arshad-ViT storage.

This is NOT a learned classifier.  It answers the exact question that was
previously conflated: how classification evidence changes as the number of
UNIQUE SOURCE PIXELS grows, while storage is capped at 1,808 bytes / 14,464 bits.

Protocol:
- official CIFAR-10
- generator-7 source-pixel walk on flattened 32x32 coordinates
- 5,13,25,41,61,85,113 UNIQUE pixels
- RGB bytes only: 3 bytes/pixel
- Z256 circular L1 MEASURE
- nearest memory among a fixed 5,000-image training subset
- labels used only after ranking for top-1 scoring
- no resize, softmax, floats, learned weights, or synthetic data
- final 113-pixel observation is packed into the corrected [16,113] byte view
  (RGB in channels 0..2; remaining channels zero) and round-tripped through
  the [128,113] bit manifold.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np
from mprc_structural.bit_manifold import pack_bytes,unpack_bytes,MANIFOLD_BITS,STORAGE_BYTES

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
COUNTS=(5,13,25,41,61,85,113)
SEED=20260927
MEMORY=5000
QUERIES=500
START=127
GEN=7

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for z in iter(lambda:f.read(1<<20),b""): h.update(z)
 return h.hexdigest()

def data_dir():
 if not ARCH.exists() or md5(ARCH)!=MD5: urllib.request.urlretrieve(URL,ARCH)
 d=CACHE/"cifar-10-batches-py"
 if not d.exists():
  with tarfile.open(ARCH,"r:gz") as tf: tf.extractall(CACHE)
 return d

def load_batch(p):
 with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
 x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
 y=np.asarray(d[b"labels"],dtype=np.int64)
 return x,y

def ringdist_sum(mem,q):
 # mem [N,P,3], q [P,3]
 d=np.abs(mem.astype(np.int16)-q.astype(np.int16)[None,...])
 d=np.minimum(d,256-d)
 return d.sum(axis=(1,2),dtype=np.int64)

d=data_dir()
tx=[];ty=[]
for i in range(1,6):
 x,y=load_batch(d/f"data_batch_{i}");tx.append(x);ty.append(y)
train_x=np.concatenate(tx);train_y=np.concatenate(ty)
test_x,test_y=load_batch(d/"test_batch")
rng=np.random.default_rng(SEED)
midx=rng.permutation(len(train_x))[:MEMORY]
qidx=rng.permutation(len(test_x))[:QUERIES]

# Exact generator-7 source coordinate sequence.
pix=np.asarray([(START+GEN*t)%1024 for t in range(113)],dtype=np.int32)
assert len(np.unique(pix))==113
rows=pix//32; cols=pix%32

metrics={}
examples=[]
for P in COUNTS:
 rr=rows[:P];cc=cols[:P]
 mem=train_x[midx][:,rr,cc,:]
 correct=0
 byclass=np.zeros(10,dtype=np.int64)
 totals=np.zeros(10,dtype=np.int64)
 for qi in qidx:
  q=test_x[qi,rr,cc,:]
  e=ringdist_sum(mem,q)
  winner=int(np.argmin(e))
  pred=int(train_y[midx[winner]]);truth=int(test_y[qi])
  correct+=int(pred==truth);totals[truth]+=1;byclass[truth]+=int(pred==truth)
 metrics[str(P)]={
   "unique_source_pixels":P,
   "source_rgb_bytes":3*P,
   "storage_capacity_bytes":STORAGE_BYTES,
   "storage_utilization":(3*P)/STORAGE_BYTES,
   "top1_accuracy":correct/len(qidx),
   "correct":correct,
   "queries":len(qidx),
   "per_class_accuracy":[(float(byclass[c]/totals[c]) if totals[c] else None) for c in range(10)]
 }

# Storage proof on final real CIFAR observations.
roundtrips=0
for qi in qidx[:128]:
 view=np.zeros((16,113),dtype=np.uint8)
 view[0]=test_x[qi,rows,cols,0]
 view[1]=test_x[qi,rows,cols,1]
 view[2]=test_x[qi,rows,cols,2]
 bits=pack_bytes(view)
 back=unpack_bytes(bits)
 assert np.array_equal(back,view)
 roundtrips+=1

R={
 "experiment":"CIFAR-10 corrected storage-vs-source-pixel audit",
 "status":"COMPLETE",
 "dataset":"official CIFAR-10",
 "memory_images":MEMORY,
 "query_images":QUERIES,
 "generator":{"start":START,"step":GEN,"domain":1024,"unique_113":True},
 "storage":{"manifold_bits":MANIFOLD_BITS,"bytes":STORAGE_BYTES,"byte_view":[16,113]},
 "final_observation":{"unique_source_pixels":113,"RGB_source_bytes":339,"unused_byte_capacity":STORAGE_BYTES-339},
 "storage_roundtrip_real_images":roundtrips,
 "metrics":metrics,
 "claim_boundary":"Counts 5..113 are UNIQUE CIFAR source pixels in this audit, not REACT support states. Retrieval is a label-blind ring-distance control, not a trained ViT classifier."
}
out=ROOT/"results"/"cifar_bitstorage_pixel_audit.json";out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(R,indent=2),encoding="utf-8")
print(json.dumps(R,indent=2))
