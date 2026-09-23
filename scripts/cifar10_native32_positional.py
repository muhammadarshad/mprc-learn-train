"""
CIFAR-10 native-32 POSITION-PRESERVING transport diagnostic.

This is the controlled follow-up to cifar10_native32.py.

Only one substantive change:
    OLD: each 7x16 / 16x7 relation field -> 16-bin histogram.
    NEW: retain every qpair state at its block-local relation position.

Frozen constraints retained:
- official CIFAR-10 train/test split;
- no resizing;
- native rectangles 7x16 and 16x7;
- partial boundary structures explicit, never vacuum padding;
- Z256/QH4 quarter states;
- no GD, no QKV, no Softmax.

This is still a Layer-D diagnostic, not full Arshad-ViT B5:
ADI-9, BIND, REACT, MEASURE, and learned metadata semantics are not added here.
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
SCORE_SCALE=1
TEST_CHUNK=500

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

def relation_fields(qblock):
    """
    Yield (relation_kind, A) where A is [N,P] uint8 and every P coordinate
    remains distinct. No histogram/pooling.
    """
    if qblock.shape[2]>=2:
        yield "H", ((qblock[:,:,:-1]<<2)|qblock[:,:,1:]).reshape(len(qblock),-1)
    if qblock.shape[1]>=2:
        yield "V", ((qblock[:,:-1,:]<<2)|qblock[:,1:,:]).reshape(len(qblock),-1)
    if qblock.shape[1]>=2 and qblock.shape[2]>=2:
        yield "D1", ((qblock[:,:-1,:-1]<<2)|qblock[:,1:,1:]).reshape(len(qblock),-1)
        yield "D2", ((qblock[:,:-1,1:]<<2)|qblock[:,1:,:-1]).reshape(len(qblock),-1)

def fit_table(A,y):
    """
    A: [N,P], categorical states 0..15.
    Returns [P,16,K] integer log-posterior table.
    """
    P=A.shape[1]
    counts=np.zeros((P,K,16),dtype=np.int32)
    offsets=(16*np.arange(P,dtype=np.int32))[None,:]

    for cls in range(K):
        R=A[y==cls].astype(np.int32,copy=False)
        codes=R+offsets
        bc=np.bincount(codes.ravel(),minlength=P*16).reshape(P,16)
        counts[:,cls,:]=bc

    total=counts.sum(axis=1,keepdims=True)
    logp=np.log((counts+ALPHA)/(total+K*ALPHA))
    return np.rint(logp*SCORE_SCALE).astype(np.int16).transpose(0,2,1)

def score_table(table,A,score):
    P=A.shape[1]
    fi=np.arange(P,dtype=np.int64)[None,:]
    for start in range(0,len(A),TEST_CHUNK):
        stop=min(len(A),start+TEST_CHUNK)
        chunk=A[start:stop]
        # [chunk,P,K] then sum over retained relation positions.
        score[start:stop]+=table[fi,chunk].sum(axis=1,dtype=np.int64)

def accuracy(score,y):
    return float(np.mean(score.argmax(1)==y))

def orientation_pass(train_a,test_a,train_y,test_y,orientation,score):
    qtr=(train_a>>6).astype(np.uint8)
    qte=(test_a>>6).astype(np.uint8)
    width=0
    table_cells=0
    block_report=[]

    for bidx,(r0,r1,c0,c1) in enumerate(blocks(orientation)):
        trb=qtr[:,r0:r1,c0:c1]
        teb=qte[:,r0:r1,c0:c1]
        one={"block":bidx,"box":[r0,r1,c0,c1],"shape":[r1-r0,c1-c0],"relations":{}}

        for kind,Atr in relation_fields(trb):
            # construct matching test relation field
            test_fields=dict(relation_fields(teb))
            Ate=test_fields[kind]
            table=fit_table(Atr,train_y)
            score_table(table,Ate,score)
            width+=Atr.shape[1]
            table_cells+=table.size
            one["relations"][kind]=int(Atr.shape[1])

        block_report.append(one)

    return width,table_cells,block_report

train_x,train_y,test_x,test_y=load_cifar()
assert train_x.shape==(50000,32,32,3)
assert test_x.shape==(10000,32,32,3)

train_ch=channels(train_x)
test_ch=channels(test_x)

names=list(train_ch.keys())
scores={
    "H":np.zeros((len(test_y),K),dtype=np.int64),
    "V":np.zeros((len(test_y),K),dtype=np.int64),
}
feature_widths={}
cumulative={}
block_reports={}
total_cells=0

for name in names:
    block_reports[name]={}
    for orient in ("H","V"):
        width,cells,report=orientation_pass(
            train_ch[name],test_ch[name],train_y,test_y,orient,scores[orient]
        )
        feature_widths[f"{name}:{orient}"]=int(width)
        total_cells+=cells
        block_reports[name][orient]=report

    cumulative[name]={
        "H":accuracy(scores["H"],test_y),
        "V":accuracy(scores["V"],test_y),
        "H_plus_V":accuracy(scores["H"]+scores["V"],test_y),
    }
    print(name,cumulative[name],flush=True)

result={
  "dataset":{
    "name":"CIFAR-10",
    "official_train":50000,
    "official_test":10000,
    "shape":[32,32,3],
    "archive_md5":MD5,
  },
  "controlled_change":"retain block-local qpair position instead of 16-bin histogram",
  "frozen_geometry":{
    "orientations":{"H":"7x16","V":"16x7"},
    "padding":0,
    "partial_structures_explicit":True,
    "H_blocks":len(blocks("H")),
    "V_blocks":len(blocks("V")),
  },
  "channels":names,
  "feature_widths":feature_widths,
  "cumulative_accuracy_after_channel":cumulative,
  "accuracy":{
    "H_only":accuracy(scores["H"],test_y),
    "V_only":accuracy(scores["V"],test_y),
    "H_plus_V":accuracy(scores["H"]+scores["V"],test_y),
  },
  "integer_lut":{
    "alpha":ALPHA,
    "score_scale":SCORE_SCALE,
    "table_dtype":"int16",
    "table_cells":int(total_cells),
    "learned_table_bytes":int(total_cells*2),
  },
  "block_reports":block_reports,
  "note":"Layer-D positional rectangular transport diagnostic; still not full Arshad-ViT B5."
}

out=ROOT/"results"/"cifar10_native32_positional.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
