"""
MPRC raw-byte Centroid -> Orbital KNN CIFAR classifier
======================================================

PRECONDITION:
  scripts/centroid_residue_orbital_knn_gate.py must PASS first.

PURPOSE
-------
Test the user's proposed readout mechanism without handcrafted vision features.

INPUT
-----
Raw CIFAR RGB bytes only: 32*32*3 = 3072 Z256 atoms.
No bit planes. No LoG/chroma/gradient feature stack. No class evidence LUT.

TRAINING
--------
For each class:
  1. Compute the site-wise circular Frechet CENTER SET under cdist:
       C_i = argmin_v sum_x cdist(v, x_i)
  2. Project that center-set product back onto the observed class examples:
       choose the training image minimizing sum_i min_{v in C_i} cdist(x_i,v)
     This gives one actual byte-vector center representative mu_c without
     inventing arithmetic averaging or pi.
  3. Arrange training examples by
       rho(x)=sum_i cdist(mu_c[i],x[i])
  4. Keep the four nearest non-center examples (K_STORE=4).

INFERENCE
---------
The stored neighbor j has four phase-complete variants:
    j_k = j + 64*k mod256, k=0..3
because
    mu + ((j-mu)+64k) = j+64k mod256.

For each query:
  - measure each of the 40 stored neighbors under all four phases;
  - retain the minimum integer ring energy and phase k;
  - take global Top-4 neighbors (K_QUERY=4);
  - majority vote class;
  - vote ties -> lower total neighbor energy -> class id host tie.

No softmax. No probability training. No gradients.

The model reports nearest-center accuracy separately to isolate the value of
the orbital KNN readout.
"""

from pathlib import Path
import hashlib, json, pickle, tarfile, urllib.request, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"
CACHE.mkdir(parents=True,exist_ok=True)

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"

K_CLASSES=10
K_STORE=4
K_QUERY=4
SEED=20260925
DIMS=32*32*3

def md5(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dataset():
    if not ARCHIVE.exists() or md5(ARCHIVE)!=MD5:
        urllib.request.urlretrieve(URL,ARCHIVE)
    assert md5(ARCHIVE)==MD5
    folder=CACHE/"cifar-10-batches-py"
    if not folder.exists():
        with tarfile.open(ARCHIVE,"r:gz") as tf:
            tf.extractall(CACHE)
    return folder

def load_batch(path):
    with open(path,"rb") as f:
        d=pickle.load(f,encoding="bytes")
    x=d[b"data"].astype(np.uint8)   # native CIFAR channel-major 3072-byte record
    y=np.asarray(d[b"labels"],dtype=np.int64)
    assert x.shape[1]==DIMS
    return x,y

def load_cifar():
    folder=ensure_dataset()
    xs=[];ys=[]
    for i in range(1,6):
        x,y=load_batch(folder/f"data_batch_{i}")
        xs.append(x);ys.append(y)
    tx=np.concatenate(xs);ty=np.concatenate(ys)
    qx,qy=load_batch(folder/"test_batch")
    return tx,ty,qx,qy

def split_40_5_5(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[];shadow=[]
    for c in range(K_CLASSES):
        ix=np.flatnonzero(y==c)
        ix=ix[rng.permutation(len(ix))]
        val.extend(ix[:500]);shadow.extend(ix[500:1000]);fit.extend(ix[1000:])
    fit=np.asarray(sorted(fit),dtype=np.int64)
    val=np.asarray(sorted(val),dtype=np.int64)
    shadow=np.asarray(sorted(shadow),dtype=np.int64)
    assert (len(fit),len(val),len(shadow))==(40000,5000,5000)
    return fit,val,shadow

# Exact cdist LUT: runtime constant, not learned state.
_v=np.arange(256,dtype=np.int16)
_diff=np.abs(_v[:,None]-_v[None,:])
CDIST=np.minimum(_diff,256-_diff).astype(np.uint8)
assert CDIST.shape==(256,256)
assert int(CDIST.max())==128

def class_center_and_neighbors(Xc, global_ix):
    """Exact integer center-set -> observed representative -> four neighbors."""
    n,d=Xc.shape
    assert d==DIMS

    # Hist[d,value].
    hist=np.zeros((d,256),dtype=np.int32)
    for i in range(d):
        hist[i]=np.bincount(Xc[:,i],minlength=256)

    # Exact Frechet L1 costs for every candidate byte at every site.
    # Max cost 4000*128=512000 fits int32.
    costs=hist @ CDIST.astype(np.int32).T
    best=costs.min(axis=1)
    is_min=(costs==best[:,None])
    tie_counts=is_min.sum(axis=1)

    # Distance from every byte value to the full minimizer set at each site.
    dset=np.empty((d,256),dtype=np.uint8)
    for i in range(d):
        C=np.flatnonzero(is_min[i])
        dset[i]=CDIST[:,C].min(axis=1)

    # Project the center set back to a real observed class example.
    # This avoids arbitrary collapse of a tied center set to one ring byte.
    proj_energy=dset[np.arange(d)[:,None],Xc.T].sum(axis=0,dtype=np.int64)
    minE=int(proj_energy.min())
    local_candidates=np.flatnonzero(proj_energy==minE)
    # Host tie only: smallest original dataset index.
    rep_local=min(local_candidates,key=lambda q:int(global_ix[q]))
    mu=Xc[rep_local].copy()
    rep_global=int(global_ix[rep_local])

    # Arrange all class examples by exact ring-L1 radius from the representative.
    radii=CDIST[Xc,mu[None,:]].sum(axis=1,dtype=np.int64)
    order=np.lexsort((global_ix,radii))
    neigh=[]
    for li in order:
        if int(li)==int(rep_local):
            continue
        neigh.append(int(li))
        if len(neigh)==K_STORE:
            break
    assert len(neigh)==K_STORE

    return {
        "center":mu,
        "center_global_index":rep_global,
        "center_set_projection_energy":minE,
        "unique_center_sites":int(np.count_nonzero(tie_counts==1)),
        "tied_center_sites":int(np.count_nonzero(tie_counts>1)),
        "max_center_set_size":int(tie_counts.max()),
        "neighbor_local_indices":neigh,
        "neighbor_global_indices":[int(global_ix[i]) for i in neigh],
        "neighbor_radii":[int(radii[i]) for i in neigh],
        "neighbors":Xc[neigh].copy(),
    }

def build_model(Xfit,yfit,global_fit_ix):
    classes=[]
    for c in range(K_CLASSES):
        mask=(yfit==c)
        Xc=Xfit[mask]
        gix=global_fit_ix[mask]
        rec=class_center_and_neighbors(Xc,gix)
        rec["class"]=c
        classes.append(rec)
        print("class",c,"center",rec["center_global_index"],
              "radii",rec["neighbor_radii"],
              "unique_sites",rec["unique_center_sites"],flush=True)

    centers=np.stack([r["center"] for r in classes]).astype(np.uint8)
    neighbors=np.concatenate([r["neighbors"] for r in classes],axis=0).astype(np.uint8)
    labels=np.repeat(np.arange(K_CLASSES,dtype=np.int64),K_STORE)

    # Four exact phase variants per stored neighbor.
    phases=np.stack([((neighbors.astype(np.uint16)+64*k)&255).astype(np.uint8)
                     for k in range(4)],axis=1)  # [40,4,D]
    return classes,centers,neighbors,labels,phases

def ring_energy_to_prototypes(X,protos,batch=64):
    """Return [N,P] exact sum-cdist energies."""
    N=len(X);P=len(protos)
    out=np.empty((N,P),dtype=np.int64)
    for st in range(0,N,batch):
        en=min(N,st+batch)
        # [B,P,D] runtime cdist LUT reads.
        d=CDIST[X[st:en,None,:],protos[None,:,:]]
        out[st:en]=d.sum(axis=2,dtype=np.int64)
    return out

def predict_centers(X,centers):
    E=ring_energy_to_prototypes(X,centers,batch=128)
    return E.argmin(axis=1),E

def predict_orbital_knn(X,phases,labels,batch=32):
    N=len(X)
    pred=np.empty(N,dtype=np.int64)
    phase_hist=np.zeros(4,dtype=np.int64)
    selected_dist_sum=0
    neighbor_vote_hist=np.zeros((K_QUERY+1,),dtype=np.int64)

    flat=phases.reshape(len(labels)*4,DIMS)

    for st in range(0,N,batch):
        en=min(N,st+batch)
        # Exact energy to all 160 phase-specific prototypes.
        E=ring_energy_to_prototypes(X[st:en],flat,batch=batch)
        E=E.reshape(en-st,len(labels),4)
        bestphase=E.argmin(axis=2)          # host tie -> lowest k; energy is tie-invariant
        bestE=E.min(axis=2)

        for bi in range(en-st):
            # Stable global Top-4 by (energy, stored-neighbor id).
            order=np.lexsort((np.arange(len(labels)),bestE[bi]))[:K_QUERY]
            labs=labels[order]
            ds=bestE[bi,order]
            ks=bestphase[bi,order]
            for k in ks:
                phase_hist[int(k)]+=1
            selected_dist_sum+=int(ds.sum())

            counts=np.bincount(labs,minlength=K_CLASSES)
            m=int(counts.max())
            winners=np.flatnonzero(counts==m)
            neighbor_vote_hist[m]+=1
            if len(winners)==1:
                pred[st+bi]=int(winners[0])
            else:
                # Tie -> smallest summed selected-neighbor energy for that class.
                best=None
                for c in winners:
                    s=int(ds[labs==c].sum())
                    key=(s,int(c))
                    if best is None or key<best[0]:
                        best=(key,int(c))
                pred[st+bi]=best[1]

    diag={
        "selected_phase_histogram":phase_hist.tolist(),
        "mean_sum_distance_top4":float(selected_dist_sum/max(1,N)),
        "max_vote_count_histogram":neighbor_vote_hist.tolist(),
    }
    return pred,diag

def metrics(y,p):
    conf=np.zeros((K_CLASSES,K_CLASSES),dtype=np.int64)
    for a,b in zip(y,p):
        conf[int(a),int(b)]+=1
    per=[float(conf[c,c]/conf[c].sum()) for c in range(K_CLASSES)]
    return {
        "accuracy":float(np.mean(y==p)),
        "correct":int(np.trace(conf)),
        "total":int(len(y)),
        "per_class_accuracy":per,
        "confusion_matrix":conf.tolist(),
    }

t0=time.time()
train_x,train_y,test_x,test_y=load_cifar()
fit_ix,val_ix,shadow_ix=split_40_5_5(train_y)

Xfit=train_x[fit_ix];yfit=train_y[fit_ix]
classes,centers,neighbors,neighbor_labels,phases=build_model(Xfit,yfit,fit_ix)

# No validation-driven model changes: evaluate the already-fixed K=4 mechanism.
sets={
    "validation":(train_x[val_ix],train_y[val_ix]),
    "shadow":(train_x[shadow_ix],train_y[shadow_ix]),
    "exploratory_test":(test_x,test_y),
}

results={}
for name,(X,y) in sets.items():
    pc,_=predict_centers(X,centers)
    pk,diag=predict_orbital_knn(X,phases,neighbor_labels)
    results[name]={
        "nearest_center":metrics(y,pc),
        "orbital_knn4":metrics(y,pk),
        "orbital_diagnostics":diag,
    }
    print(name,
          "center",results[name]["nearest_center"]["accuracy"],
          "orbital",results[name]["orbital_knn4"]["accuracy"],
          flush=True)

# Serialize compact model diagnostics, not the full byte vectors in JSON.
class_meta=[]
for r in classes:
    class_meta.append({
        "class":r["class"],
        "center_global_index":r["center_global_index"],
        "center_set_projection_energy":r["center_set_projection_energy"],
        "unique_center_sites":r["unique_center_sites"],
        "tied_center_sites":r["tied_center_sites"],
        "max_center_set_size":r["max_center_set_size"],
        "neighbor_global_indices":r["neighbor_global_indices"],
        "neighbor_radii":r["neighbor_radii"],
    })

model_bytes=int(centers.nbytes+neighbors.nbytes+neighbor_labels.nbytes)

report={
    "model":"MPRC-RawRGB-Centroid-Orbital-KNN4",
    "discipline":{
        "required_gate":"centroid_residue_orbital_knn_gate.py",
        "workflow_runs_gate_before_training":True,
        "validation_used_for_selection":False,
        "byte_atomicity":"raw CIFAR RGB bytes only",
    },
    "dataset":{
        "fit":40000,"validation":5000,"shadow":5000,"official_test":10000,
        "official_test_status":"exploratory; prior experiments exposed this split",
        "dims":DIMS,"archive_md5":MD5,
    },
    "centroid":{
        "method":"site-wise circular Frechet minimizer set under cdist, projected to nearest observed class example",
        "arithmetic_mean":False,
        "pi_or_trig":False,
        "class_metadata":class_meta,
    },
    "neighbors":{
        "K_store":K_STORE,
        "arrangement":"sum cdist from class center representative",
        "stored_neighbor_count":int(len(neighbors)),
    },
    "inference":{
        "phase_offsets":[0,64,128,192],
        "same_global_phase_per_neighbor_vector":True,
        "K_query":K_QUERY,
        "vote":"majority; tie by summed orbital energy then class id",
        "softmax":False,
        "probability_layer":False,
    },
    "model_state_bytes":model_bytes,
    "runtime_constant_cdist_bytes":int(CDIST.nbytes),
    "results":results,
    "runtime_seconds":time.time()-t0,
    "claim_boundary":"cdist metric and orbital code properties are gated. CIFAR accuracy, K=4 usefulness, projected-center choice, and raw-RGB class separability are empirical."
}

out=ROOT/"results"/"mprc_rawrgb_centroid_orbital_knn4.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
