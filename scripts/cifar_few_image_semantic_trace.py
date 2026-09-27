"""Few-image CIFAR semantic trace through the frozen ViT pipeline.

Diagnostic only. No training and no architecture changes.

For one deterministic query per CIFAR-10 class and a balanced memory of
100 images/class, rank candidates label-blind at these checkpoints:

S0 raw RGB Z256 distance
S1 16 deterministic encoder/projection channels
S2 full joint directional ADI9 relation distance across 16 channels
S3 v29 relational IDENTIFY: QH4/J2 pattern agreement then ADI energy
S4 frozen BIND->REACT_7->MEASURE, applied only to the best 32 S3 candidates

After ranking, reveal labels and report:
- top1 class
- first same-class rank
- same-class count in top10
- same-class vs wrong-class margin
- per-channel contributions for the best same/wrong candidates

The point is to locate where semantic class neighbourhood is created or lost.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

from mprc_structural.identify import qh4_pattern_type
from mprc_structural.cifar_frame import frame_rgb32,to_manifolds
from mprc_structural.batch_attention import batch_energy

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
NCLASS=10; MEM_PER_CLASS=100; SEED=20260928
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]
CR=np.arange(2,30,4,dtype=np.int16); CC=np.arange(2,30,4,dtype=np.int16)
ARMS=[(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2)]

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
 return np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy(),np.asarray(d[b"labels"],dtype=np.int64)
def load():
 d=ensure();xs=[];ys=[]
 for i in range(1,6):
  x,y=lb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
 tx=np.concatenate(xs);ty=np.concatenate(ys);vx,vy=lb(d/"test_batch")
 return tx,ty,vx,vy
def rd(a,b):
 d=np.abs(a.astype(np.int16)-b.astype(np.int16));return np.minimum(d,256-d)
def luma(x):
 r=x[...,0].astype(np.uint16);g=x[...,1].astype(np.uint16);b=x[...,2].astype(np.uint16)
 return ((77*r+150*g+29*b)>>8).astype(np.uint8)
def blur(a,r):
 p=np.pad(a,((0,0),(r,r),(r,r)),mode="edge").astype(np.uint32)
 I=np.pad(p,((0,0),(1,0),(1,0)),mode="constant").cumsum(1,dtype=np.uint32).cumsum(2,dtype=np.uint32)
 k=2*r+1;s=I[:,k:,k:]-I[:,:-k,k:]-I[:,k:,:-k]+I[:,:-k,:-k]
 return ((s+k*k//2)//(k*k)).astype(np.uint8)
def fields(x):
 R=x[...,0];G=x[...,1];B=x[...,2];Y=luma(x)
 Gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
 Chroma=(x.max(-1).astype(np.int16)-x.min(-1).astype(np.int16)).astype(np.uint8)
 gx=np.zeros_like(Y,dtype=np.int16);gy=np.zeros_like(Y,dtype=np.int16)
 gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
 gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)
 Gx=np.clip(128+gx//2,0,255).astype(np.uint8);Gy=np.clip(128+gy//2,0,255).astype(np.uint8)
 Grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)
 lap=np.zeros_like(Y,dtype=np.int16);c=Y[:,1:-1,1:-1].astype(np.int16)
 lap[:,1:-1,1:-1]=Y[:,:-2,1:-1].astype(np.int16)+Y[:,2:,1:-1].astype(np.int16)+Y[:,1:-1,:-2].astype(np.int16)+Y[:,1:-1,2:].astype(np.int16)-4*c
 Lap=np.clip(128+lap//4,0,255).astype(np.uint8)
 b1=blur(Y,1);b2=blur(Y,2);b4=blur(Y,4);b8=blur(Y,8)
 H1=np.abs(Y.astype(np.int16)-b1.astype(np.int16)).astype(np.uint8)
 H2=np.abs(b1.astype(np.int16)-b2.astype(np.int16)).astype(np.uint8)
 M4=np.abs(b2.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 L8=np.abs(b4.astype(np.int16)-b8.astype(np.int16)).astype(np.uint8)
 Contrast=np.abs(Y.astype(np.int16)-b4.astype(np.int16)).astype(np.uint8)
 dgy=np.zeros_like(Y,dtype=np.int16);dgx=np.zeros_like(Y,dtype=np.int16)
 dgy[:,:,1:-1]=gy[:,:,2:]-gy[:,:,:-2];dgx[:,1:-1,:]=gx[:,2:,:]-gx[:,:-2,:]
 Curl=np.clip(128+(dgy-dgx)//4,0,255).astype(np.uint8)
 return [R,G,B,Gray,Y,Chroma,Gx,Gy,Grad,Lap,H1,H2,M4,L8,Contrast,Curl]
def adi(a):
 rr=CR[:,None];cc=CC[None,:];C=a[:,rr,cc].astype(np.uint16);out=np.empty((len(a),7,7,9),dtype=np.uint8);tot=C.copy();vals=[]
 for dr,dc in ARMS:
  v=a[:,rr+dr,cc+dc].astype(np.uint16);vals.append(v);tot=(tot+v)&255
 out[:,:,:,0]=tot.astype(np.uint8)
 for j,v in enumerate(vals,1):out[:,:,:,j]=((C-v)&255).astype(np.uint8)
 return out
def qh4_types(A):
 # A [N,16,7,7,9], return small integer type code per descriptor; -1 invalid.
 N=A.shape[0];out=np.full((N,16,7,7),-1,dtype=np.int8)
 m={"anchor":0,"vacuum-adjacent":1,"gate-cluster":2,"step-column":3,"cross-quarter":4,"quarter-stripe":5}
 for n in range(N):
  for ch in range(16):
   for r in range(7):
    for c in range(7):
     t=qh4_pattern_type(tuple(map(int,A[n,ch,r,c])))
     if t is not None:out[n,ch,r,c]=m[t]
 return out
def metrics(order,labels,truth,score):
 labs=labels[order];same=np.flatnonzero(labs==truth)
 first=int(same[0]+1) if len(same) else None
 top10=int((labs[:10]==truth).sum())
 same_scores=score[labels==truth];wrong_scores=score[labels!=truth]
 # lower is better for all scalar energy stages
 return {"top1":NAMES[int(labs[0])],"top1_same":bool(labs[0]==truth),"first_same_rank":first,"same_in_top10":top10,
         "best_same_energy":int(same_scores.min()) if len(same_scores) else None,
         "best_wrong_energy":int(wrong_scores.min()) if len(wrong_scores) else None,
         "margin_wrong_minus_same":int(wrong_scores.min()-same_scores.min()) if len(same_scores) else None}

tx,ty,vx,vy=load();rng=np.random.default_rng(SEED)
mem_ids=[]
for c in range(10):
 z=np.flatnonzero(ty==c);z=z[rng.permutation(len(z))];mem_ids.extend(z[:MEM_PER_CLASS])
mem_ids=np.asarray(mem_ids,dtype=np.int64)
# one deterministic query per class
q_ids=np.asarray([np.flatnonzero(vy==c)[0] for c in range(10)],dtype=np.int64)
M=tx[mem_ids];ML=ty[mem_ids];Q=vx[q_ids];QL=vy[q_ids]

MF=fields(M);QF=fields(Q)
M16=np.stack(MF,axis=1);Q16=np.stack(QF,axis=1)
MADI=np.stack([adi(f) for f in MF],axis=1);QADI=np.stack([adi(f) for f in QF],axis=1)
MT=qh4_types(MADI);QT=qh4_types(QADI)

rows=[]
stage_top1={"S0_RGB":0,"S1_16CH":0,"S2_ADI9":0,"S3_IDENTIFY":0,"S4_ATTN_TOP32":0}
for qi in range(10):
 truth=int(QL[qi]);q=Q[qi]
 # S0
 e0=rd(M,q[None]).sum((1,2,3),dtype=np.int64);o0=np.argsort(e0,kind="stable")
 # S1
 per1=rd(M16,Q16[qi][None]).sum((2,3),dtype=np.int64) # [M,16]
 e1=per1.sum(1,dtype=np.int64);o1=np.argsort(e1,kind="stable")
 # S2 full joint ADI distance
 per2=rd(MADI,QADI[qi][None]).sum((2,3,4),dtype=np.int64) # [M,16]
 e2=per2.sum(1,dtype=np.int64);o2=np.argsort(e2,kind="stable")
 # S3 lexicographic identify. Pattern type agreement primary, ADI energy secondary.
 qtyp=QT[qi]
 valid=(qtyp[None]>=0)&(MT>=0)
 pm=((MT==qtyp[None])&valid).sum((1,2,3),dtype=np.int64)
 # convert lexicographic pair to deterministic order; scalar shown remains ADI energy.
 o3=np.lexsort((np.arange(len(M)),e2,-pm))
 # S4 frozen attention only on top32 IDENTIFY candidates, raw RGB manifolds.
 top=o3[:32]
 qman=to_manifolds(frame_rgb32(q))
 cman=np.stack([to_manifolds(frame_rgb32(M[j])) for j in top],axis=0)
 ea=batch_energy(cman,qman,lut=None,rounds=7)
 oa=np.argsort(ea,kind="stable");att_order=top[oa]
 # for comparable metric vector set infinity outside selected set
 e4=np.full(len(M),np.iinfo(np.int64).max,dtype=np.int64);e4[top]=ea
 # channel diagnostics best same/wrong at S1/S2
 bs=int(np.where(ML==truth)[0][np.argmin(e1[ML==truth])]);bw=int(np.where(ML!=truth)[0][np.argmin(e1[ML!=truth])])
 row={"query_index":int(q_ids[qi]),"truth":NAMES[truth],
      "S0_RGB":metrics(o0,ML,truth,e0),
      "S1_16CH":metrics(o1,ML,truth,e1),
      "S2_ADI9":metrics(o2,ML,truth,e2),
      "S3_IDENTIFY":metrics(o3,ML,truth,e2),
      "S4_ATTN_TOP32":metrics(oa,ML[top],truth,ea),
      "identify":{"top_pattern_matches":int(pm[o3[0]]),"best_same_pattern_matches":int(pm[ML==truth].max()),"best_wrong_pattern_matches":int(pm[ML!=truth].max())},
      "encoder_contribution_best_same":{CHANNEL_NAMES[k]:int(per1[bs,k]) for k in range(16)},
      "encoder_contribution_best_wrong":{CHANNEL_NAMES[k]:int(per1[bw,k]) for k in range(16)},
      "adi_contribution_best_same":{CHANNEL_NAMES[k]:int(per2[bs,k]) for k in range(16)},
      "adi_contribution_best_wrong":{CHANNEL_NAMES[k]:int(per2[bw,k]) for k in range(16)}}
 for st in stage_top1:
  if row[st]["top1_same"]:stage_top1[st]+=1
 rows.append(row)
 print(truth,NAMES[truth],[(s,rows[-1][s]["top1"],rows[-1][s]["first_same_rank"]) for s in stage_top1],flush=True)

R={"experiment":"few-image semantic trace","memory":{"images":len(M),"per_class":MEM_PER_CLASS},"queries":10,
   "stage_top1_same":stage_top1,"rows":rows,
   "interpretation":"A stage is useful for classification only if same-class neighbours move toward the front consistently before labels are used."}
out=ROOT/"results"/"cifar_few_image_semantic_trace.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps({"stage_top1_same":stage_top1,"summary":[{"truth":r["truth"],**{k:{"top1":r[k]["top1"],"rank":r[k]["first_same_rank"]} for k in stage_top1}} for r in rows]},indent=2))
