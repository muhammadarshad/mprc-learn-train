"""CIFAR-10 MPRC raw-RGB memory classifier — first post-survival training.

ALL architecture/routing rules are frozen by no-label gates before this script runs.

INPUT
-----
Raw CIFAR RGB bytes only. No learned encoder, no bit-plane collapse, no
handcrafted LoG/Curl feature bank.

LOCAL
-----
For each of the three source byte channels, sample the fixed 7x7 source
anchor lattice. Each anchor is the proved directional ADI-9 state:
    C,U1,U2,D1,D2,F1,F2,B1,B2 -> Lambda + 8 deltas in Z256.

SELECT
------
Position/displacement-compatible exact multiset intersection:
    M_n(Q)=sum_D min(c_Q(D),c_n(D)).
Keep ALL memories with maximum M_n. If max=0, abstain.

REACTION LUT
------------
Fit the precommitted v28 label-free ring-medoid ReactionLUT from the 40k
fit observations in generator-transported execution coordinates.

READOUT
-------
For every selected memory, sum exact BIND->REACT_7->MEASURE energy over
the three RGB manifolds. Keep all minimum-energy memories.
If their labels agree, predict that class. If labels disagree, abstain.

No Top-K, threshold, radius, class weighting, softmax, QKV or GD.
"""

from __future__ import annotations

from pathlib import Path
import hashlib, json, pickle, tarfile, urllib.request, time
import numpy as np

from mprc_structural.directional_adi import INV9
from mprc_structural.occurrence_select import SortedOccurrenceSelectIndex
from mprc_structural.cifar_frame import (
    OBS_H,OBS_W,SRC_H,SRC_W,AY,AX,FILL,frame_rgb32,to_manifolds
)
from mprc_structural.batch_attention import transport_batch,batch_energy
from mprc_structural.reaction_lut import fit_from_counts

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"
CACHE.mkdir(parents=True,exist_ok=True)

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"
K=10
SEED=20260927

CENTER_ROWS=np.arange(2,30,4,dtype=np.int16)
CENTER_COLS=np.arange(2,30,4,dtype=np.int16)
ARMS=(
    (-1,0),(-2,0),
    (1,0),(2,0),
    (0,1),(0,2),
    (0,-1),(0,-2),
)
assert len(CENTER_ROWS)==len(CENTER_COLS)==7
ANCHORS=49
FEATURES=3*ANCHORS

def md5(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()

def ensure_dataset():
    if not ARCHIVE.exists() or md5(ARCHIVE)!=MD5:
        urllib.request.urlretrieve(URL,ARCHIVE)
    assert md5(ARCHIVE)==MD5
    d=CACHE/"cifar-10-batches-py"
    if not d.exists():
        with tarfile.open(ARCHIVE,"r:gz") as tf:
            tf.extractall(CACHE)
    return d

def load_batch(path):
    with open(path,"rb") as f:
        d=pickle.load(f,encoding="bytes")
    x=d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.uint8)
    y=np.asarray(d[b"labels"],dtype=np.int64)
    return x,y

def load_cifar():
    d=ensure_dataset()
    xs=[];ys=[]
    for i in range(1,6):
        x,y=load_batch(d/f"data_batch_{i}")
        xs.append(x);ys.append(y)
    tx=np.concatenate(xs);ty=np.concatenate(ys)
    vx,vy=load_batch(d/"test_batch")
    return tx,ty,vx,vy

def split_40_5_5(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[];shadow=[]
    for c in range(K):
        ix=np.flatnonzero(y==c)
        ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500])
        shadow.extend(ix[500:1000])
        fit.extend(ix[1000:])
    fit=np.asarray(sorted(fit),dtype=np.int64)
    val=np.asarray(sorted(val),dtype=np.int64)
    shadow=np.asarray(sorted(shadow),dtype=np.int64)
    assert (len(fit),len(val),len(shadow))==(40000,5000,5000)
    assert np.bincount(y[fit],minlength=K).tolist()==[4000]*K
    assert np.bincount(y[val],minlength=K).tolist()==[500]*K
    assert np.bincount(y[shadow],minlength=K).tolist()==[500]*K
    return fit,val,shadow

def descriptors_rgb(x):
    """[N,32,32,3] -> [N,147,9] directional ADI bytes."""
    x=np.asarray(x)
    if x.ndim!=4 or x.shape[1:]!=(32,32,3) or x.dtype!=np.uint8:
        raise ValueError("expected uint8 [N,32,32,3]")
    rr=CENTER_ROWS[:,None]
    cc=CENTER_COLS[None,:]
    out=np.empty((len(x),3,7,7,9),dtype=np.uint8)
    for ch in range(3):
        a=x[:,:,:,ch]
        C=a[:,rr,cc].astype(np.uint16)
        total=C.copy()
        vals=[]
        for dr,dc in ARMS:
            v=a[:,rr+dr,cc+dc].astype(np.uint16)
            vals.append(v)
            total=(total+v)&255
        out[:,ch,:,:,0]=total.astype(np.uint8)
        for j,v in enumerate(vals,start=1):
            out[:,ch,:,:,j]=((C-v)&255).astype(np.uint8)
    return out.reshape(len(x),FEATURES,9)

def frame_batch(rgb):
    """[B,32,32,3] -> [B,3,128,113] exact observed manifolds."""
    x=np.asarray(rgb)
    B=len(x)
    obs=np.full((B,3,OBS_H,OBS_W),FILL,dtype=np.uint8)
    obs[:,:,AY:AY+SRC_H,AX:AX+SRC_W]=np.transpose(x,(0,3,1,2))
    return np.transpose(obs,(0,1,3,2)).copy()

def fit_reaction_lut(rgb_fit,batch=128):
    """Use every fit observation; labels are not passed into this function."""
    counts=np.zeros((256,256),dtype=np.int64)
    observations=0
    for st in range(0,len(rgb_fit),batch):
        en=min(len(rgb_fit),st+batch)
        man=frame_batch(rgb_fit[st:en])
        logical=transport_batch(man)
        for slab in (0,64):
            lo=slab; hi=slab+64
            s=logical[:,:,lo:hi,:]
            center=s[:,:,1:-1,1:-1].astype(np.uint16)
            staple=(
                center
                +s[:,:,:-2,1:-1].astype(np.uint16)
                +s[:,:,2:,1:-1].astype(np.uint16)
                +s[:,:,1:-1,:-2].astype(np.uint16)
                +s[:,:,1:-1,2:].astype(np.uint16)
            )&255
            code=staple.astype(np.int64)*256+center.astype(np.int64)
            counts += np.bincount(code.ravel(),minlength=65536).reshape(256,256)
            observations += int(code.size)
        if st and (st//batch)%50==0:
            print("LUT observations",st,"/",len(rgb_fit),flush=True)
    assert int(counts.sum())==observations
    lut=fit_from_counts(counts)
    return lut,counts,observations

def evaluate(name,x,y,index,fit_x,fit_y,lut,energy_batch_size=64):
    total=len(x)
    correct=0
    predicted=0
    no_support=0
    mixed_energy_tie=0
    support_sum=0
    selected_sum=0
    max_selected=0
    max_support_seen=0
    support_hist={}
    selected_hist={}
    conf=np.zeros((K,K),dtype=np.int64)

    desc=descriptors_rgb(x)

    for qi in range(total):
        sel,scores=index.select(desc[qi])
        best=int(scores.max()) if len(scores) else 0
        max_support_seen=max(max_support_seen,best)
        support_hist[str(best)]=support_hist.get(str(best),0)+1

        if len(sel)==0:
            no_support+=1
            continue

        support_sum+=best
        selected_sum+=len(sel)
        max_selected=max(max_selected,len(sel))
        # Coarse histogram prevents JSON explosion.
        bucket=(
            "1" if len(sel)==1 else
            "2-4" if len(sel)<=4 else
            "5-16" if len(sel)<=16 else
            "17-64" if len(sel)<=64 else
            "65-256" if len(sel)<=256 else
            ">256"
        )
        selected_hist[bucket]=selected_hist.get(bucket,0)+1

        qman=to_manifolds(frame_rgb32(x[qi]))
        energies=np.empty(len(sel),dtype=np.int64)
        for st in range(0,len(sel),energy_batch_size):
            en=min(len(sel),st+energy_batch_size)
            ids=sel[st:en]
            cman=frame_batch(fit_x[ids])
            energies[st:en]=batch_energy(cman,qman,lut=lut,rounds=7)

        emin=int(energies.min())
        tied=sel[energies==emin]
        labels=np.unique(fit_y[tied])
        if len(labels)!=1:
            mixed_energy_tie+=1
            continue

        pred=int(labels[0])
        predicted+=1
        truth=int(y[qi])
        conf[truth,pred]+=1
        correct+=int(pred==truth)

        if (qi+1)%250==0:
            print(name,qi+1,"/",total,
                  "coverage",predicted/(qi+1),
                  "acc_all",correct/(qi+1),
                  flush=True)

    return {
        "total":total,
        "predicted":predicted,
        "abstained":total-predicted,
        "no_exact_support":no_support,
        "mixed_label_min_energy_tie":mixed_energy_tie,
        "coverage":predicted/total,
        "accuracy_all":correct/total,
        "accuracy_covered":(correct/predicted if predicted else None),
        "correct":correct,
        "mean_max_support_on_selected":(support_sum/(total-no_support) if total!=no_support else 0.0),
        "mean_selected_candidates_on_selected":(selected_sum/(total-no_support) if total!=no_support else 0.0),
        "max_selected_candidates":max_selected,
        "max_support_seen":max_support_seen,
        "max_support_histogram":support_hist,
        "selected_candidate_histogram":selected_hist,
        "confusion_matrix_predicted_only":conf.tolist(),
    }

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx,shadow_idx=split_40_5_5(train_y)

xfit=train_x[fit_idx]
yfit=train_y[fit_idx]

# Integration exactness before labels influence any learned state.
for i in range(128):
    obs=frame_rgb32(xfit[i])
    assert np.array_equal(obs[:,AY:AY+32,AX:AX+32],np.transpose(xfit[i],(2,0,1)))
    assert np.array_equal(
        descriptors_rgb(xfit[i:i+1])[0],
        descriptors_rgb(xfit[i:i+1])[0]
    )

print("fit ReactionLUT (label-free)",flush=True)
lut,counts,lut_observations=fit_reaction_lut(xfit)

print("build exact occurrence memory",flush=True)
memory_desc=descriptors_rgb(xfit)
index=SortedOccurrenceSelectIndex(memory_desc)
del memory_desc

print("evaluate validation",flush=True)
val=evaluate("val",train_x[val_idx],train_y[val_idx],index,xfit,yfit,lut)
print("evaluate shadow",flush=True)
shadow=evaluate("shadow",train_x[shadow_idx],train_y[shadow_idx],index,xfit,yfit,lut)
print("evaluate official test (exploratory)",flush=True)
test=evaluate("test",test_x,test_y,index,xfit,yfit,lut)

result={
    "model":"MPRC-CIFAR-RGB-post-survival-v0",
    "status":"EMPIRICAL TRAINING RESULT",
    "survival_contract":{
        "parent_pretraining_contract":"PASS required by workflow",
        "typed_arshad_block":"PASS required by workflow",
        "occurrence_select":"PASS required by workflow",
        "cifar_frame":"PASS required by workflow",
        "batch_attention":"PASS required by workflow",
        "reaction_lut_gate":"PASS required by workflow"
    },
    "input":{
        "source":"raw CIFAR RGB bytes",
        "derived_feature_channels":False,
        "bit_planes":False,
        "resize":False,
        "observation_frame":[3,113,128],
        "execution_manifolds":[3,128,113],
        "frame_fill":int(FILL)
    },
    "local":{
        "context":["C","U1","U2","D1","D2","F1","F2","B1","B2"],
        "descriptor":["Lambda","dU1","dU2","dD1","dD2","dF1","dF2","dB1","dB2"],
        "anchors_per_channel":ANCHORS,
        "channels":3,
        "descriptors_per_image":FEATURES,
        "sampling_status":"experimental CIFAR32 source lattice"
    },
    "training":{
        "fit_images":40000,
        "validation_images":5000,
        "shadow_images":5000,
        "official_test_images":10000,
        "reaction_lut_observations":lut_observations,
        "reaction_lut":list(map(int,lut)),
        "reaction_lut_nonidentity_entries":int(np.count_nonzero(lut!=np.arange(256,dtype=np.uint8))),
        "labels_used_for_routing":False,
        "labels_used_for_reaction_lut":False,
        "gradient_descent":False,
        "softmax":False,
        "top_k":False
    },
    "selection":{
        "rule":"M_n(Q)=sum_D min(c_Q(D),c_n(D)); all max survive; zero => abstain",
        "position_permutation_invariant":True
    },
    "readout":{
        "energy":"sum of three exact BIND->REACT_7->MEASURE channel energies",
        "minimum_energy_ties":"predict only if all minimum-energy memories share one class; else abstain"
    },
    "validation":val,
    "shadow_holdout":shadow,
    "official_test":{
        "status":"exploratory continuity only; historical test has been exposed",
        **test
    },
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"First classifier after full math/code survival. Raw RGB only. Exact-match routing may be too sparse; any failure is retained rather than repaired with post-hoc thresholds or Top-K."
}

out=ROOT/"results"/"mprc_cifar_rgb_postsurvival_v0.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
np.save(ROOT/"results"/"mprc_cifar_rgb_postsurvival_v0_lut.npy",lut)
print(json.dumps(result,indent=2),flush=True)
