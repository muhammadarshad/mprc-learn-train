"""v31 — CIFAR feature-channel specialization audit.

Diagnostic only. No architecture change.

Hypothesis
----------
Arshad's ViT evaluates features in parallel. The 16 deterministic channels
should therefore be inspected independently, not collapsed by equal summation.

For each channel:
1) classify validation images using RAW 1+8 factorized evidence only;
2) report standalone accuracy;
3) on 10 fixed queries, measure whether same-class neighbourhood beats
   wrong-class neighbourhood in that channel;
4) report which channels rescue queries that RGB/global sums miss.

Labels are used only for post-hoc diagnostic scoring, never to alter channel
definitions or query ranking.
"""
from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
K=10;SEED=20260928
NAMES=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CHANNEL_NAMES=["R","G","B","Gray","Luma","Chroma","Gx","Gy","Grad","Laplacian","H1","H2","M4","L8","Contrast","Curl"]
CR=np.arange(2,30,4,dtype=np.int16);CC=np.arange(2,30,4,dtype=np.int16)
ARMS=[(-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2)]

def md5(p):
 h=hashlib.md5()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def ensure():
 if not ARCH.exists() or md5(ARCH)!=MD5:urllib.request.urlretrieve(URL,ARCH)
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
def raw9(a):
 rr=CR[:,None];cc=CC[None,:];parts=[a[:,rr,cc]]
 for dr,dc in ARMS:parts.append(a[:,rr+dr,cc+dc])
 return np.stack(parts,-1).reshape(len(a),49*9)
def evidence(A,y):
 F=A.shape[1];cnt=np.zeros((F,K,256),dtype=np.int32);off=(256*np.arange(F,dtype=np.int64))[None,:]
 for c in range(K):
  z=A[y==c].astype(np.int64,copy=False)
  cnt[:,c,:]=np.bincount((z+off).ravel(),minlength=F*256).reshape(F,256)
 total=cnt.sum(1,keepdims=True)
 return (K*cnt-total).astype(np.int16).transpose(0,2,1)
def score(T,A):
 fi=np.arange(A.shape[1],dtype=np.int64)[None,:];S=np.zeros((len(A),K),dtype=np.int64)
 for st in range(0,len(A),500):S[st:st+500]=T[fi,A[st:st+500]].sum(1,dtype=np.int64)
 return S
def rd(a,b):
 d=np.abs(a.astype(np.int16)-b.astype(np.int16));return np.minimum(d,256-d)

tx,ty,vx,vy=load();rng=np.random.default_rng(SEED)
# 40k/5k split
fit=[];val=[]
for c in range(K):
 ix=np.flatnonzero(ty==c);ix=ix[rng.permutation(len(ix))]
 val.extend(ix[:500]);fit.extend(ix[1000:])
fit=np.asarray(sorted(fit));val=np.asarray(sorted(val));yf=ty[fit];yv=ty[val]

F=fields(tx);VF=fields(vx)
standalone={}
val_scores=[]
for k,name in enumerate(CHANNEL_NAMES):
 A=raw9(F[k]);T=evidence(A[fit],yf);S=score(T,A[val]);acc=float(np.mean(S.argmax(1)==yv))
 standalone[name]=acc;val_scores.append(S)
 print(name,acc,flush=True)

# Combinations chosen diagnostically from validation only.
# Greedy forward addition: add channel only if validation accuracy increases.
selected=[];cur=np.zeros_like(val_scores[0]);cur_acc=.1;path=[]
remaining=list(range(16))
while remaining:
 best=None
 for k in remaining:
  s=cur+val_scores[k];a=float(np.mean(s.argmax(1)==yv))
  key=(a,-k)
  if best is None or key>best[0]:best=(key,k,s)
 if best[0][0] <= cur_acc:break
 _,k,s=best;selected.append(k);remaining.remove(k);cur=s;cur_acc=best[0][0]
 path.append({"channel":CHANNEL_NAMES[k],"validation_accuracy":cur_acc})

# Few-image specialist audit: balanced 100/class memory.
mem=[]
for c in range(K):
 z=np.flatnonzero(ty==c);z=z[rng.permutation(len(z))];mem.extend(z[:100])
mem=np.asarray(mem);ML=ty[mem]
qids=np.asarray([np.flatnonzero(vy==c)[0] for c in range(K)])
query_rows=[]
specialist_counts={n:0 for n in CHANNEL_NAMES}
for qi,qid in enumerate(qids):
 truth=int(vy[qid]);row={"truth":NAMES[truth],"channels":{}}
 for k,name in enumerate(CHANNEL_NAMES):
  M=raw9(F[k][mem]);Q=raw9(VF[k][qid:qid+1])[0]
  e=rd(M,Q[None]).sum(1,dtype=np.int64)
  same=e[ML==truth];wrong=e[ML!=truth]
  bs=int(same.min());bw=int(wrong.min());margin=bw-bs
  o=np.argsort(e,kind="stable");top=int(ML[o[0]])
  useful=margin>0
  specialist_counts[name]+=int(useful)
  row["channels"][name]={"top1":NAMES[top],"top1_same":top==truth,"margin_wrong_minus_same":int(margin),
                          "first_same_rank":int(np.flatnonzero(ML[o]==truth)[0]+1)}
 query_rows.append(row)

R={"experiment":"v31 feature-channel specialization audit",
   "standalone_validation_accuracy":standalone,
   "greedy_validation_path":path,
   "greedy_selected_channels":[CHANNEL_NAMES[k] for k in selected],
   "greedy_final_validation_accuracy":cur_acc,
   "few_image_specialist_queries_won":specialist_counts,
   "queries":query_rows,
   "claim_boundary":"Diagnostic only. Greedy selection is not promoted architecture; it measures whether useful feature channels are specialized and whether equal all-channel aggregation hides them."}
out=ROOT/"results"/"v31_feature_channel_specialization.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps({"standalone":standalone,"greedy_path":path,"specialists":specialist_counts},indent=2))
