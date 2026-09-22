"""
CIFAR-10 native-32 transport diagnostic for MPRC learning.

Basis:
- Official CIFAR-10 train/test split (50,000 / 10,000).
- No image resizing.
- No 8x8 byte windows.
- Rectangular byte structures are 7x16 and 16x7, matching the frozen
  Arshad-ViT orientation transport shapes.
- QH4 observation state is the ordered pair of quarter states (16 categories).
- Learned state is categorical occupancy/evidence in LUTs. No gradient update,
  QKV, softmax, or learned convolution is used.

This is a Layer-D learning/transport diagnostic, not yet the full B5 model.
It does not claim G11 multires packing: partial 32-pixel boundary structures
are processed explicitly and reported rather than silently padded.
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

def luma(rgb):
    r=rgb[...,0].astype(np.uint16)
    g=rgb[...,1].astype(np.uint16)
    b=rgb[...,2].astype(np.uint16)
    return ((77*r+150*g+29*b)>>8).astype(np.uint8)

def channels(rgb):
    Y=luma(rgb)
    gx=np.zeros_like(Y,dtype=np.int16)
    gy=np.zeros_like(Y,dtype=np.int16)
    gx[:,:,1:-1]=Y[:,:,2:].astype(np.int16)-Y[:,:,:-2].astype(np.int16)
    gy[:,1:-1,:]=Y[:,2:,:].astype(np.int16)-Y[:,:-2,:].astype(np.int16)
    grad=np.clip((np.abs(gx)+np.abs(gy))//2,0,255).astype(np.uint8)
    gyu=np.clip(128+gy//2,0,255).astype(np.uint8)
    return {
      "R":rgb[...,0],
      "G":rgb[...,1],
      "B":rgb[...,2],
      "Y":Y,
      "GY":gyu,
      "GRAD":grad,
    }

def blocks(orientation):
    """
    Explicit partial structures; no padding.
    H: 7 rows x 16 cols (last row group has height 4)
    V: 16 rows x 7 cols (last col group has width 4)
    """
    out=[]
    if orientation=="H":
        for r0 in range(0,32,7):
            r1=min(32,r0+7)
            for c0 in range(0,32,16):
                out.append((r0,r1,c0,c0+16))
    elif orientation=="V":
        for r0 in range(0,32,16):
            for c0 in range(0,32,7):
                c1=min(32,c0+7)
                out.append((r0,r0+16,c0,c1))
    else:
        raise ValueError(orientation)
    return out

def block_hist_features(a,orientation):
    """
    a: [N,32,32] uint8
    Per block and edge orientation (H,V,D1,D2), emit 16 qpair-bin counts.
    Count values fit uint8 for these rectangles.
    """
    q=(a>>6).astype(np.uint8)
    feats=[]
    for r0,r1,c0,c1 in blocks(orientation):
        z=q[:,r0:r1,c0:c1]
        rels=[]
        if z.shape[2]>=2:
            rels.append((z[:,:,:-1]<<2)|z[:,:,1:])
        if z.shape[1]>=2:
            rels.append((z[:,:-1,:]<<2)|z[:,1:,:])
        if z.shape[1]>=2 and z.shape[2]>=2:
            rels.append((z[:,:-1,:-1]<<2)|z[:,1:,1:])
            rels.append((z[:,:-1,1:]<<2)|z[:,1:,:-1])
        for rel in rels:
            flat=rel.reshape(len(a),-1)
            for cat in range(16):
                feats.append((flat==cat).sum(axis=1).astype(np.uint8))
    return np.stack(feats,axis=1)

K=10
ALPHA=0.05

class LUT:
    def fit(self,A,y):
        maxstate=int(A.max())+1
        cnt=np.zeros((A.shape[1],K,maxstate),dtype=np.int32)
        for c in range(K):
            R=A[y==c]
            for j in range(A.shape[1]):
                cnt[j,c]=np.bincount(R[:,j],minlength=maxstate)
        total=cnt.sum(axis=1,keepdims=True)
        self.tab=np.log((cnt+ALPHA)/(total+K*ALPHA)).astype(np.float32)
        return self
    def score(self,A):
        S=np.zeros((len(A),K),dtype=np.float32)
        for j in range(A.shape[1]):
            S+=np.take(self.tab[j],A[:,j],axis=1).T
        return S

def accuracy(S,y):
    return float(np.mean(S.argmax(1)==y))

train_x,train_y,test_x,test_y=load_cifar()
assert train_x.shape==(50000,32,32,3)
assert test_x.shape==(10000,32,32,3)

train_ch=channels(train_x)
test_ch=channels(test_x)

names=list(train_ch.keys())
scores={"H":np.zeros((len(test_y),K),dtype=np.float32),
        "V":np.zeros((len(test_y),K),dtype=np.float32)}
feature_widths={}
state_bytes=0

for name in names:
    for orient in ("H","V"):
        Atr=block_hist_features(train_ch[name],orient)
        Ate=block_hist_features(test_ch[name],orient)
        feature_widths[f"{name}:{orient}"]=int(Atr.shape[1])
        model=LUT().fit(Atr,train_y)
        scores[orient]+=model.score(Ate)
        state_bytes+=model.tab.nbytes
        del Atr,Ate,model

result={
  "dataset":{
    "name":"CIFAR-10",
    "official_train":50000,
    "official_test":10000,
    "shape":[32,32,3],
    "archive_md5":MD5,
  },
  "frozen_geometry":{
    "orientations":{"H":"7x16","V":"16x7"},
    "padding":0,
    "partial_structures_explicit":True,
    "H_blocks":len(blocks("H")),
    "V_blocks":len(blocks("V")),
    "H_partials":[list(x) for x in blocks("H") if (x[1]-x[0])!=7],
    "V_partials":[list(x) for x in blocks("V") if (x[3]-x[2])!=7],
  },
  "channels":names,
  "feature_widths":feature_widths,
  "accuracy":{
    "H_only":accuracy(scores["H"],test_y),
    "V_only":accuracy(scores["V"],test_y),
    "H_plus_V":accuracy(scores["H"]+scores["V"],test_y),
  },
  "learned_table_bytes_float32":int(state_bytes),
  "note":"Layer-D rectangular transport diagnostic; not full Arshad-ViT B5."
}

out=ROOT/"results"/"cifar10_native32_transport.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
