"""
CIFAR-10 structural position-metadata experiment.

Research question:
Can the owner's 16-channel + 1-polarity structural interpretation carry
position explicitly, instead of hiding absolute position in classifier indices?

For each native rectangular block:
    - preserve source pixels untouched;
    - derive 16 channel observations;
    - store ONE local position per channel;
    - store ONE separate block polarity category.

No image resizing. No QKV. No Softmax. No learned convolution.

This is a METADATA-CODEC HYPOTHESIS test, not a frozen codec.
The exact 112x112 canonical transport identities are implemented separately in
mprc_structural.block_local112.

Three label-blind selectors are compared on a validation subset only:
    byte_max    : largest uint8 channel state
    ring_radius : largest distance from singularity 128
    adi8        : largest 8-neighbour circular differential energy

The official CIFAR test split is evaluated only after selector choice.
"""

from pathlib import Path
import hashlib, json, pickle, tarfile, urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"
CACHE.mkdir(parents=True,exist_ok=True)

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"

K=10
ALPHA=0.05
SEED=20260923
CHANNEL_NAMES=(
    "R","G","B","gray","luma","chroma",
    "gx","gy","grad","lap",
    "h1","h2","m4","l8","contrast","orient",
)
SELECTORS=("byte_max","ring_radius","adi8")
# In a position-only codec, the orientation row stores where orientation is
# most reliable. That location is selected from grad; no floating angle is
# inserted into the ring path.
ORIENT_SOURCE="grad"

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
    data=d[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.uint8)
    labels=np.asarray(d[b"labels"],dtype=np.int64)
    return data,labels

def load_cifar():
    folder=ensure_dataset()
    xs=[];ys=[]
    for i in range(1,6):
        x,y=load_batch(folder/f"data_batch_{i}")
        xs.append(x);ys.append(y)
    train_x=np.concatenate(xs); train_y=np.concatenate(ys)
    test_x,test_y=load_batch(folder/"test_batch")
    return train_x,train_y,test_x,test_y

def split_fit_val(y):
    rng=np.random.default_rng(SEED)
    fit=[];val=[]
    for cls in range(K):
        ids=np.flatnonzero(y==cls)
        rng.shuffle(ids)
        # CIFAR-10 has exactly 5000 training samples/class.
        val.extend(ids[:500])
        fit.extend(ids[500:])
    return np.asarray(sorted(fit),dtype=np.int64),np.asarray(sorted(val),dtype=np.int64)

def box_blur_batch(a,r):
    """Integer edge-padded box blur for [N,H,W] uint8."""
    if r==0:
        return a.copy()
    k=2*r+1
    p=np.pad(a.astype(np.uint32),((0,0),(r,r),(r,r)),mode="edge")
    # summed-area table with leading zero row/column
    s=np.pad(p,((0,0),(1,0),(1,0)),mode="constant")
    s=s.cumsum(axis=1,dtype=np.uint32).cumsum(axis=2,dtype=np.uint32)
    out=(s[:,k:,k:]-s[:,:-k,k:]-s[:,k:,:-k]+s[:,:-k,:-k])//(k*k)
    return out.astype(np.uint8)

def channels15(rgb):
    R=rgb[...,0]
    G=rgb[...,1]
    B=rgb[...,2]
    gray=((R.astype(np.uint16)+G.astype(np.uint16)+B.astype(np.uint16))//3).astype(np.uint8)
    luma=((77*R.astype(np.uint16)+150*G.astype(np.uint16)+29*B.astype(np.uint16))>>8).astype(np.uint8)
    chroma=(np.maximum.reduce([R,G,B])-np.minimum.reduce([R,G,B])).astype(np.uint8)

    gx_s=np.zeros_like(luma,dtype=np.int16)
    gy_s=np.zeros_like(luma,dtype=np.int16)
    gx_s[:,:,1:-1]=luma[:,:,2:].astype(np.int16)-luma[:,:,:-2].astype(np.int16)
    gy_s[:,1:-1,:]=luma[:,2:,:].astype(np.int16)-luma[:,:-2,:].astype(np.int16)

    gx=np.clip(128+gx_s//2,0,255).astype(np.uint8)
    gy=np.clip(128+gy_s//2,0,255).astype(np.uint8)
    grad=np.clip((np.abs(gx_s)+np.abs(gy_s))//2,0,255).astype(np.uint8)

    lap_s=np.zeros_like(luma,dtype=np.int16)
    c=luma[:,1:-1,1:-1].astype(np.int16)
    lap_s[:,1:-1,1:-1]=(
        luma[:,:-2,1:-1].astype(np.int16)
        +luma[:,2:,1:-1].astype(np.int16)
        +luma[:,1:-1,:-2].astype(np.int16)
        +luma[:,1:-1,2:].astype(np.int16)
        -4*c
    )
    lap=np.clip(128+lap_s//4,0,255).astype(np.uint8)

    b1=box_blur_batch(luma,1)
    b2=box_blur_batch(luma,2)
    b4=box_blur_batch(luma,4)
    b8=box_blur_batch(luma,8)

    def adiff(a,b):
        return np.abs(a.astype(np.int16)-b.astype(np.int16)).astype(np.uint8)

    h1=adiff(luma,b1)
    h2=adiff(b1,b2)
    m4=adiff(b2,b4)
    l8=adiff(b4,b8)
    contrast=adiff(luma,b4)

    return {
        "R":R,"G":G,"B":B,"gray":gray,"luma":luma,"chroma":chroma,
        "gx":gx,"gy":gy,"grad":grad,"lap":lap,
        "h1":h1,"h2":h2,"m4":m4,"l8":l8,"contrast":contrast,
    }

def blocks(shape):
    """Canonical-first partial structures for 32x32."""
    out=[]
    if shape=="16x7":
        for r0 in range(0,32,16):
            r1=min(32,r0+16)
            for c0 in range(0,32,7):
                c1=min(32,c0+7)
                out.append((r0,r1,c0,c1))
    elif shape=="7x16":
        for r0 in range(0,32,7):
            r1=min(32,r0+7)
            for c0 in range(0,32,16):
                c1=min(32,c0+16)
                out.append((r0,r1,c0,c1))
    else:
        raise ValueError(shape)
    return out

def cdist_u8(a,b):
    d=np.abs(a.astype(np.int16)-b.astype(np.int16))
    return np.minimum(d,256-d).astype(np.uint16)

def select_flat(block,selector):
    """Return raw flat index in the actual partial block for every sample."""
    n,h,w=block.shape
    flat=block.reshape(n,-1)

    if selector=="byte_max":
        return flat.argmax(axis=1).astype(np.int32)

    if selector=="ring_radius":
        strength=np.abs(flat.astype(np.int16)-128)
        return strength.argmax(axis=1).astype(np.int32)

    if selector=="adi8":
        # Exact ring circular differential energy on the 8 neighbours.
        # Borders are excluded; every CIFAR partial structure is >=4 in each dim.
        center=block[:,1:-1,1:-1]
        energy=np.zeros(center.shape,dtype=np.uint16)
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==0 and dc==0:
                    continue
                neigh=block[:,1+dr:h-1+dr,1+dc:w-1+dc]
                energy+=cdist_u8(center,neigh)

        eflat=energy.reshape(n,-1)
        q=eflat.argmax(axis=1)
        inner_w=w-2
        rr=(q//inner_w)+1
        cc=(q%inner_w)+1
        return (rr*w+cc).astype(np.int32)

    raise ValueError(selector)

def canonical_local_index(raw_index,h,w,shape):
    r=raw_index//w
    c=raw_index%w
    if shape=="16x7":
        # Canonical local stride is 7 even when the last block is partial.
        return (7*r+c).astype(np.uint8)
    if shape=="7x16":
        # Canonical local stride is 16 even when the last block is partial.
        return (16*r+c).astype(np.uint8)
    raise ValueError(shape)

def polarity_code(v):
    # Experimental categorical encoding only:
    # 0 = <128, 1 = singularity 128, 2 = >128.
    return np.where(v<128,0,np.where(v>128,2,1)).astype(np.uint8)

def metadata_features(rgb,selector,chunk=2500):
    """Return [N, 2 phases * 10 blocks * (16 positions + 1 polarity)]."""
    n=len(rgb)
    phases=("16x7","7x16")
    width=sum(len(blocks(s))*17 for s in phases)
    out=np.empty((n,width),dtype=np.uint8)

    for start in range(0,n,chunk):
        stop=min(n,start+chunk)
        ch=channels15(rgb[start:stop])
        col=0

        for shape in phases:
            for r0,r1,c0,c1 in blocks(shape):
                h=r1-r0
                w=c1-c0
                selected_raw={}

                for name in CHANNEL_NAMES:
                    source_name=ORIENT_SOURCE if name=="orient" else name
                    block=ch[source_name][:,r0:r1,c0:c1]
                    raw=select_flat(block,selector)
                    selected_raw[name]=raw
                    out[start:stop,col]=canonical_local_index(raw,h,w,shape)
                    col+=1

                # One separate polarity state per block, referenced to luma's
                # selected location.  This is deliberately reported as a
                # hypothesis, not a frozen semantic.
                luma_block=ch["luma"][:,r0:r1,c0:c1].reshape(stop-start,-1)
                raw=selected_raw["luma"]
                vals=luma_block[np.arange(stop-start),raw]
                out[start:stop,col]=polarity_code(vals)
                col+=1

        assert col==width

    return out

def fit_table(A,y,maxstate=112):
    F=A.shape[1]
    cnt=np.zeros((F,K,maxstate),dtype=np.int32)
    offsets=(maxstate*np.arange(F,dtype=np.int32))[None,:]
    for cls in range(K):
        R=A[y==cls].astype(np.int32,copy=False)
        codes=R+offsets
        bc=np.bincount(codes.ravel(),minlength=F*maxstate).reshape(F,maxstate)
        cnt[:,cls,:]=bc
    total=cnt.sum(axis=1,keepdims=True)
    return np.log((cnt+ALPHA)/(total+K*ALPHA)).astype(np.float32)

def score(tab,A):
    S=np.zeros((len(A),K),dtype=np.float32)
    for j in range(A.shape[1]):
        S+=np.take(tab[j],A[:,j],axis=1).T
    return S

def accuracy(S,y):
    return float(np.mean(S.argmax(1)==y))

train_x,train_y,test_x,test_y=load_cifar()
fit_idx,val_idx=split_fit_val(train_y)

features_train={}
features_test={}
validation={}

for selector in SELECTORS:
    print("building",selector,flush=True)
    Atr=metadata_features(train_x,selector)
    features_train[selector]=Atr

    tab=fit_table(Atr[fit_idx],train_y[fit_idx])
    va=accuracy(score(tab,Atr[val_idx]),train_y[val_idx])
    validation[selector]=va
    print(selector,"validation",va,flush=True)

# Validation-only selector choice. Tie priority is ring-native first.
priority={"ring_radius":2,"adi8":1,"byte_max":0}
selected=max(SELECTORS,key=lambda s:(validation[s],priority[s]))
print("selected",selected,flush=True)

Ate=metadata_features(test_x,selected)
features_test[selected]=Ate
Atr=features_train[selected]
tab=fit_table(Atr,train_y)

# Fixed phase widths: 10 blocks * 17 metadata states = 170 each.
phase_width=170
S16=score(tab[:phase_width],Ate[:,:phase_width])
S7=score(tab[phase_width:],Ate[:,phase_width:])
Sall=S16+S7

result={
    "dataset":{
        "name":"CIFAR-10",
        "official_train":50000,
        "official_test":10000,
        "shape":[32,32,3],
        "archive_md5":MD5,
    },
    "metadata_hypothesis":{
        "channel_rows":16,
        "polarity_dimensions":1,
        "channels":list(CHANNEL_NAMES),
        "orient_position_source":ORIENT_SOURCE,
        "per_block":"16 local-position states + 1 polarity category",
        "pixels_replaced":False,
        "position_categories":"canonical local index 0..111",
        "polarity_categories":{"0":"<128","1":"=128","2":">128"},
    },
    "geometry":{
        "phase_16x7_blocks":len(blocks("16x7")),
        "phase_7x16_blocks":len(blocks("7x16")),
        "partial_structures_explicit":True,
        "padding":0,
        "features_per_phase":phase_width,
        "features_total":int(Atr.shape[1]),
    },
    "validation":{
        "fit_samples":int(len(fit_idx)),
        "validation_samples":int(len(val_idx)),
        "selector_accuracy":validation,
        "selected_selector":selected,
    },
    "official_test_accuracy":{
        "phase_16x7":accuracy(S16,test_y),
        "phase_7x16":accuracy(S7,test_y),
        "combined":accuracy(Sall,test_y),
    },
    "learned_table":{
        "alpha":ALPHA,
        "dtype":"float32",
        "cells":int(tab.size),
        "bytes":int(tab.nbytes),
    },
    "status":"EMPIRICAL METADATA-CODEC HYPOTHESIS; selector and polarity semantics are not frozen."
}

out=ROOT/"results"/"cifar10_position_metadata.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
