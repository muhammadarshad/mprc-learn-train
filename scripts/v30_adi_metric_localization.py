"""v30 — ADI metric localization on CIFAR.

No frozen equation is changed.

Question:
Directional ADI9 is an exact bijection, but is circular-L1 in ADI coordinates
a valid similarity metric for visual classification?

Compare, on identical observations:
A raw9 relation energy
B ADI9 coordinate energy
C ADI9 -> exact inverse -> raw9 energy (must equal A)
D QH4/J2 structural type agreement + raw9 relation energy

If A/C preserve class neighbourhood while B destroys it, ADI remains an
IDENTIFY coordinate/type transform but ADI-coordinate L1 is rejected as a
classification similarity metric.

No training, no labels in ranking.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np
from mprc_structural.identify import qh4_pattern_type
from mprc_structural.directional_adi import encode as adi_encode, decode as adi_decode

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
SEED=20260928; MEM_PER_CLASS=200
CR=np.arange(2,30,4,dtype=np.int16);CC=np.arange(2,30,4,dtype=np.int16)
ARMS=[(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2)]

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def ensure():
 if not ARCH.exists() or md5(ARCH)!=MD5: urllib.request.urlretrieve(URL,ARCH)
 d=CACHE/"cifar-10-batches-py"
 if not d.exists():
  with tarfile.open(ARCH,"r:gz") as tf:tf.extractall(CACHE)
 return d
def lb(p):
 with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
 return np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy(),np.asarray(d[b"labels"],dtype=np.int64)
def load():
 d=ensure();xs=[];ys=[]
 for i in range(1,6):
  x,y=lb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
 tx=np.concatenate(xs);ty=np.concatenate(ys);vx,vy=lb(d/"test_batch")
 return tx,ty,vx,vy
def rd(a,b):
 d=np.abs(a.astype(np.int16)-b.astype(np.int16));return np.minimum(d,256-d)
def raw9(x):
 rr=CR[:,None];cc=CC[None,:];parts=[x[:,rr,cc]]
 for dr,dc in ARMS:parts.append(x[:,rr+dr,cc+dc])
 return np.stack(parts,-1).reshape(len(x),49,9)
def adi_batch(r):
 N,A,_=r.shape;out=np.empty_like(r)
 for n in range(N):
  for a in range(A):
   out[n,a]=np.asarray(adi_encode(tuple(map(int,r[n,a]))),dtype=np.uint8)
 return out
def inverse_batch(z):
 N,A,_=z.shape;out=np.empty_like(z)
 for n in range(N):
  for a in range(A):
   out[n,a]=np.asarray(adi_decode(tuple(map(int,z[n,a]))),dtype=np.uint8)
 return out
def types(z):
 N,A,_=z.shape;out=np.full((N,A),-1,dtype=np.int8)
 mp={"anchor":0,"vacuum-adjacent":1,"gate-cluster":2,"step-column":3,"cross-quarter":4,"quarter-stripe":5}
 for n in range(N):
  for a in range(A):
   t=qh4_pattern_type(tuple(map(int,z[n,a])))
   if t is not None:out[n,a]=mp[t]
 return out
def m(order,labels,truth):
 labs=labels[order];same=np.flatnonzero(labs==truth)
 return {"top1":NAMES[int(labs[0])],"top1_same":bool(labs[0]==truth),
         "first_same_rank":int(same[0]+1) if len(same) else None,
         "same_in_top10":int((labs[:10]==truth).sum())}

tx,ty,vx,vy=load();rng=np.random.default_rng(SEED)
mem=[]
for c in range(10):
 z=np.flatnonzero(ty==c);z=z[rng.permutation(len(z))];mem.extend(z[:MEM_PER_CLASS])
mem=np.asarray(mem);M=tx[mem];ML=ty[mem]
qids=np.asarray([np.flatnonzero(vy==c)[0] for c in range(10)])
Q=vx[qids];QL=vy[qids]

# raw RGB only to isolate coordinate transform effect.
MR=np.concatenate([raw9(M[:,:,:,c]) for c in range(3)],axis=1) # [M,147,9]
QR=np.concatenate([raw9(Q[:,:,:,c]) for c in range(3)],axis=1)
MA=adi_batch(MR); QA=adi_batch(QR)
REC_M=inverse_batch(MA);REC_Q=inverse_batch(QA)
assert np.array_equal(REC_M,MR) and np.array_equal(REC_Q,QR)

MT=types(MA);QT=types(QA)
rows=[];wins={"RAW9":0,"ADI9":0,"RECOVERED_RAW9":0,"QH4_PLUS_RAW":0}
for i in range(10):
 truth=int(QL[i])
 eraw=rd(MR,QR[i][None]).sum((1,2),dtype=np.int64)
 eadi=rd(MA,QA[i][None]).sum((1,2),dtype=np.int64)
 erec=rd(REC_M,REC_Q[i][None]).sum((1,2),dtype=np.int64)
 assert np.array_equal(eraw,erec)

 valid=(MT>=0)&(QT[i][None]>=0)
 pm=((MT==QT[i][None])&valid).sum(1,dtype=np.int64)

 oraw=np.argsort(eraw,kind="stable")
 oadi=np.argsort(eadi,kind="stable")
 orec=np.argsort(erec,kind="stable")
 oq=np.lexsort((np.arange(len(M)),eraw,-pm))

 row={"truth":NAMES[truth],"RAW9":m(oraw,ML,truth),"ADI9":m(oadi,ML,truth),
      "RECOVERED_RAW9":m(orec,ML,truth),"QH4_PLUS_RAW":m(oq,ML,truth),
      "best_same":{"raw":int(eraw[ML==truth].min()),"adi":int(eadi[ML==truth].min())},
      "best_wrong":{"raw":int(eraw[ML!=truth].min()),"adi":int(eadi[ML!=truth].min())},
      "q_h4_valid_descriptors":int((QT[i]>=0).sum())}
 for k in wins:wins[k]+=int(row[k]["top1_same"])
 rows.append(row)
 print(row,flush=True)

# random-pair metric distortion diagnostic
ids=rng.integers(0,len(M),size=(2000,2))
rawD=[];adiD=[]
for a,b in ids:
 rawD.append(int(rd(MR[a],MR[b]).sum()))
 adiD.append(int(rd(MA[a],MA[b]).sum()))
rawD=np.asarray(rawD);adiD=np.asarray(adiD)
# rank correlation without scipy
ro=np.argsort(np.argsort(rawD,kind="stable"),kind="stable")
ao=np.argsort(np.argsort(adiD,kind="stable"),kind="stable")
rho=float(np.corrcoef(ro,ao)[0,1])

R={"experiment":"v30 ADI metric localization","top1_same":wins,"queries":rows,
   "metric":{"random_pairs":len(ids),"raw_vs_adi_rank_correlation":rho,
             "adi_inverse_exact":True},
   "decision_rule":"If ADI9 neighbourhood is materially worse than RAW9 while inverse is exact, reject ADI-coordinate circular-L1 as visual similarity; retain ADI as exact IDENTIFY/type coordinates."}
out=ROOT/"results"/"v30_adi_metric_localization.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps(R,indent=2))
