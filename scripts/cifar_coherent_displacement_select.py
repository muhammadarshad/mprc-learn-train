"""CIFAR coherent-displacement SELECT integration.

Purpose
-------
This is the first empirical classifier after the coherent-displacement SELECT
gate. It changes SELECT only; all frozen algebra remains unchanged.

Pipeline
--------
raw CIFAR RGB
-> directional ADI9 on fixed 7x7 source anchor lattice
-> lossless exact-occurrence candidate generation
-> coherent-displacement SELECT
-> frozen 128x113 / Z64 BIND -> REACT_7 -> MEASURE
-> minimum-energy label readout

No Top-K, no learned threshold, no softmax, no QKV, no change to frozen
manifold, transport, REACT, or MEASURE.

The exact-occurrence index is used only as a lossless candidate generator:
every sample sharing at least one exact ADI9 descriptor with the query survives.
Coherent displacement, not occurrence count, performs SELECT.
"""
from __future__ import annotations

from pathlib import Path
import hashlib,json,pickle,tarfile,urllib.request,time
import numpy as np

from mprc_structural.occurrence_select import SortedOccurrenceSelectIndex,_flat_keys
from mprc_structural.coherent_displacement_select import coherent_displacement_scores
from mprc_structural.cifar_frame import OBS_H,OBS_W,SRC_H,SRC_W,AY,AX,FILL,frame_rgb32,to_manifolds
from mprc_structural.batch_attention import transport_batch,batch_energy
from mprc_structural.reaction_lut import fit_from_counts

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"; CACHE.mkdir(parents=True,exist_ok=True)
URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCH=CACHE/"cifar-10-python.tar.gz"; MD5="c58f30108f718f92721af3b95e74349a"
K=10; SEED=20260927
CENTER_ROWS=np.arange(2,30,4,dtype=np.int16)
CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
ARMS=((-1,0),(-2,0),(1,0),(2,0),(0,1),(0,2),(0,-1),(0,-2))
CH=3; H=W=7; A=CH*H*W

# Runtime-bounded decision gate. Fixed before labels are read.
N_VAL=500
N_SHADOW=500

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

def loadb(p):
    with open(p,"rb") as f:d=pickle.load(f,encoding="bytes")
    x=np.asarray(d[b"data"],dtype=np.uint8).reshape(-1,3,32,32).transpose(0,2,3,1).copy()
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def load():
    d=ensure(); xs=[];ys=[]
    for i in range(1,6):
        x,y=loadb(d/f"data_batch_{i}");xs.append(x);ys.append(y)
    tx=np.concatenate(xs);ty=np.concatenate(ys)
    vx,vy=loadb(d/"test_batch")
    return tx,ty,vx,vy

def split(y):
    rng=np.random.default_rng(SEED); fit=[];val=[];shadow=[]
    for c in range(K):
        ix=np.flatnonzero(y==c);ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500]);shadow.extend(ix[500:1000]);fit.extend(ix[1000:])
    return np.asarray(sorted(fit)),np.asarray(sorted(val)),np.asarray(sorted(shadow))

def descriptors(x):
    rr=CENTER_ROWS[:,None];cc=CENTER_COLS[None,:]
    out=np.empty((len(x),CH,H,W,9),dtype=np.uint8)
    for ch in range(CH):
        a=x[:,:,:,ch]; C=a[:,rr,cc].astype(np.uint16); total=C.copy(); vals=[]
        for dr,dc in ARMS:
            v=a[:,rr+dr,cc+dc].astype(np.uint16);vals.append(v);total=(total+v)&255
        out[:,ch,:,:,0]=total.astype(np.uint8)
        for j,v in enumerate(vals,1):out[:,ch,:,:,j]=((C-v)&255).astype(np.uint8)
    return out

def frame_batch(rgb):
    x=np.asarray(rgb);B=len(x)
    obs=np.full((B,3,OBS_H,OBS_W),FILL,dtype=np.uint8)
    obs[:,:,AY:AY+SRC_H,AX:AX+SRC_W]=np.transpose(x,(0,3,1,2))
    return np.transpose(obs,(0,1,3,2)).copy()

def fit_lut(rgb,batch=128):
    counts=np.zeros((256,256),dtype=np.int64)
    for st in range(0,len(rgb),batch):
        man=frame_batch(rgb[st:st+batch]);logical=transport_batch(man)
        for slab in (0,64):
            s=logical[:,:,slab:slab+64,:]
            center=s[:,:,1:-1,1:-1].astype(np.uint16)
            staple=(center+s[:,:,:-2,1:-1].astype(np.uint16)+s[:,:,2:,1:-1].astype(np.uint16)+s[:,:,1:-1,:-2].astype(np.uint16)+s[:,:,1:-1,2:].astype(np.uint16))&255
            code=staple.astype(np.int64)*256+center.astype(np.int64)
            counts += np.bincount(code.ravel(),minlength=65536).reshape(256,256)
    return fit_from_counts(counts)

def candidate_samples(idx,qflat):
    keys=np.unique(_flat_keys(qflat))
    chunks=[]
    for key in keys:
        lo=int(np.searchsorted(idx.keys,key,side="left"))
        hi=int(np.searchsorted(idx.keys,key,side="right"))
        if hi>lo: chunks.append(idx.samples[lo:hi])
    if not chunks:return np.empty(0,dtype=np.int64)
    return np.unique(np.concatenate(chunks)).astype(np.int64)

def evaluate(name,x,y,mem_desc,idx,xfit,yfit,lut):
    desc=descriptors(x)
    correct=predicted=no_candidate=no_coherent=mixed=0
    cand_counts=[]; coherent_counts=[]; supports=[]; disps=[]
    for qi in range(len(x)):
        q=desc[qi]
        cand=candidate_samples(idx,q.reshape(A,9))
        cand_counts.append(len(cand))
        if len(cand)==0:
            no_candidate+=1;continue

        score,disp=coherent_displacement_scores(q,mem_desc[cand])
        mx=int(score.max()) if len(score) else 0
        if mx==0:
            no_coherent+=1;continue
        keep_local=np.flatnonzero(score==mx)
        selected=cand[keep_local]
        coherent_counts.append(len(selected));supports.append(mx)
        # retain displacement P as diagnostic
        disps.extend([tuple(map(int,z)) for z in disp[keep_local]])

        qman=to_manifolds(frame_rgb32(x[qi]))
        energies=np.empty(len(selected),dtype=np.int64)
        for st in range(0,len(selected),64):
            ids=selected[st:st+64]
            energies[st:st+len(ids)]=batch_energy(frame_batch(xfit[ids]),qman,lut=lut,rounds=7)
        emin=int(energies.min())
        tied=selected[energies==emin]
        labs=np.unique(yfit[tied])
        if len(labs)!=1:
            mixed+=1;continue
        pred=int(labs[0]);predicted+=1;correct+=int(pred==int(y[qi]))
        if (qi+1)%100==0:
            print(name,qi+1,"acc_all",correct/(qi+1),"coverage",predicted/(qi+1),flush=True)

    from collections import Counter
    dh=Counter(map(str,disps))
    return {
      "total":len(x),"predicted":predicted,"correct":correct,
      "accuracy_all":correct/len(x),"accuracy_covered":correct/predicted if predicted else None,
      "coverage":predicted/len(x),"no_occurrence_candidate":no_candidate,
      "no_coherent_support":no_coherent,"mixed_label_min_energy_tie":mixed,
      "candidate_count_mean":float(np.mean(cand_counts)) if cand_counts else 0.0,
      "coherent_selected_mean":float(np.mean(coherent_counts)) if coherent_counts else 0.0,
      "coherent_support_mean":float(np.mean(supports)) if supports else 0.0,
      "displacement_histogram":dict(dh)
    }

t0=time.time()
tx,ty,testx,testy=load();fit_idx,val_idx,shadow_idx=split(ty)
xfit=tx[fit_idx];yfit=ty[fit_idx]
print("build descriptors",flush=True)
mem_desc=descriptors(xfit)
idx=SortedOccurrenceSelectIndex(mem_desc.reshape(len(mem_desc),A,9))
print("fit label-free ReactionLUT",flush=True)
lut=fit_lut(xfit)

# Fixed deterministic balanced subsets from the already-frozen val/shadow partitions.
rng=np.random.default_rng(SEED+404)
def balanced_take(ix,n):
    per=n//K;out=[]
    for c in range(K):
        z=ix[ty[ix]==c]; z=z[rng.permutation(len(z))]; out.extend(z[:per])
    return np.asarray(sorted(out),dtype=np.int64)

v=balanced_take(val_idx,N_VAL);s=balanced_take(shadow_idx,N_SHADOW)
print("evaluate validation",flush=True)
vr=evaluate("val",tx[v],ty[v],mem_desc,idx,xfit,yfit,lut)
print("evaluate shadow",flush=True)
sr=evaluate("shadow",tx[s],ty[s],mem_desc,idx,xfit,yfit,lut)

R={
 "model":"CIFAR coherent-displacement SELECT integration",
 "frozen":{"manifold":[128,113],"slabs":[64,113],"generator":7,"transport_modulus":64,
           "attention":"BIND->REACT_7->MEASURE","local":"directional ADI9"},
 "selection":{"candidate_generation":"all memory samples sharing >=1 exact ADI9 descriptor",
              "decision":"max coherent common-displacement support; all ties survive",
              "labels_used":False,"top_k":False,"threshold":False},
 "validation":vr,"shadow":sr,
 "fit_images":len(fit_idx),"validation_subset":len(v),"shadow_subset":len(s),
 "runtime_seconds":time.time()-t0,
 "claim_boundary":"Decision gate for integrating the already-survived coherent-displacement SELECT. No frozen equation is changed."
}
out=ROOT/"results"/"cifar_coherent_displacement_select.json";out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(R,indent=2))
print(json.dumps(R,indent=2),flush=True)
