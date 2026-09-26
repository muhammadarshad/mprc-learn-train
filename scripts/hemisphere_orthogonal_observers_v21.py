"""v21 — Hemisphere-aware orthogonal visual observers.

v20 diagnosed OBSERVE/IDENTIFY as the bottleneck:
    shape correct 235/480
    color correct 240/480
    constructed pair 124/480

Two MPRC corrections
====================

1. SHAPE is a positive magnitude/geometric factor.
   Do NOT span 0..255 and then apply circular distance, because 0 and 255 are
   adjacent on Z256.

   Encode all geometric magnitudes in ONE hemisphere:
       0..127

   Shape state uses no resizing:
   - bbox aspect ratios
   - foreground fill
   - 16 row-band occupancies
   - 7 column-band occupancies
   - 16 left/right boundary profiles
   - 7 top/bottom boundary profiles

   The 16/7 structure is deliberate and uses integer ratios only.

2. COLOR is directional/relative, not raw brightness.
   Use signed channel relations:
       R-G, G-B, B-R

   Compress each signed difference by /2 into [-127,127], then encode around
   MPRC origin 128:

       state = 128 - signed_relation

   Therefore:
       positive relation -> <128
       negative relation -> >128

   Add quartiles/median/mean plus sign-dominance counts.

Benchmark
=========
Same real Fruits-360 2x2 cross:
    citrus+orange, citrus+green, pepper+orange, pepper+green

For each fold remove one WHOLE combination.
Resolved shape/color LUTs retain the orthogonal factors through the other
combinations.

No learned classifier. No whole-pair parameters.

    OBSERVE
      -> IDENTIFY shape state / color state
      -> MEASURE each factor LUT
      -> SELECT shape / color
      -> BIND selected factors
      -> CONSTRUCT omitted pair

Whole-pattern memory control remains for contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

MAX_TRAIN_PER_CLASS=160
MAX_TEST_PER_CLASS=120

CLASSES={
    "Orange 1":        ("round_citrus","orange"),
    "Limes 1":         ("round_citrus","green"),
    "Pepper Orange 1": ("pepper","orange"),
    "Pepper Green 1":  ("pepper","green"),
}


def cdist(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def energy(a,b)->int:
    return int(cdist(a,b).sum(dtype=np.int64))


def ratio127(num:int,den:int)->int:
    if den<=0:
        return 0
    return int((int(num)*127 + den//2)//den)


def load_rgb(p:Path)->np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8)


def paths(root:Path,split:str,cls:str,limit:int):
    d=root/split/cls
    xs=sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    return xs[:limit]


def foreground(rgb):
    return np.any(rgb<245,axis=2)


def bbox(mask):
    ys,xs=np.where(mask)
    if len(xs)==0:
        return 0,0,mask.shape[1],mask.shape[0]
    x0=int(xs.min()); x1=int(xs.max())+1
    y0=int(ys.min()); y1=int(ys.max())+1
    return x0,y0,x1,y1


def band_occupancy127(z:np.ndarray,axis:int,bands:int):
    idxs=np.array_split(np.arange(z.shape[axis]),bands)
    out=[]
    for ids in idxs:
        part=z[ids,:] if axis==0 else z[:,ids]
        out.append(ratio127(int(part.sum()),int(part.size)))
    return out


def row_boundaries127(z:np.ndarray,bands:int=16):
    groups=np.array_split(np.arange(z.shape[0]),bands)
    W=z.shape[1]
    out=[]
    den=max(1,W-1)

    for ids in groups:
        band=z[ids,:]
        cols=np.where(np.any(band,axis=0))[0]
        if len(cols)==0:
            out.extend([0,0])
        else:
            out.extend([
                ratio127(int(cols.min()),den),
                ratio127(int(cols.max()),den),
            ])
    return out


def col_boundaries127(z:np.ndarray,bands:int=7):
    groups=np.array_split(np.arange(z.shape[1]),bands)
    H=z.shape[0]
    out=[]
    den=max(1,H-1)

    for ids in groups:
        band=z[:,ids]
        rows=np.where(np.any(band,axis=1))[0]
        if len(rows)==0:
            out.extend([0,0])
        else:
            out.extend([
                ratio127(int(rows.min()),den),
                ratio127(int(rows.max()),den),
            ])
    return out


def shape_state(rgb):
    m=foreground(rgb)
    x0,y0,x1,y1=bbox(m)
    z=m[y0:y1,x0:x1]

    h,w=z.shape
    mx=max(h,w,1)

    out=[
        ratio127(w,mx),
        ratio127(h,mx),
        ratio127(int(z.sum()),int(z.size)),
    ]
    out+=band_occupancy127(z,0,16)
    out+=band_occupancy127(z,1,7)
    out+=row_boundaries127(z,16)
    out+=col_boundaries127(z,7)

    q=np.asarray(out,dtype=np.uint8)
    assert len(q)==72
    assert int(q.max())<=127
    return q


def trunc_div2_signed(x:np.ndarray)->np.ndarray:
    x=np.asarray(x,dtype=np.int16)
    pos=x>=0
    out=np.empty_like(x,dtype=np.int16)
    out[pos]=x[pos]//2
    out[~pos]=-((-x[~pos])//2)
    return out


def signed_quantile(x:np.ndarray,num:int,den:int)->int:
    x=np.asarray(x,dtype=np.int16).reshape(-1)
    if len(x)==0:
        return 0
    k=((len(x)-1)*num)//den
    return int(np.partition(x,k)[k])


def chroma_encode(signed_value:int)->int:
    v=max(-127,min(127,int(signed_value)))
    return (128-v)&255


def color_state(rgb):
    m=foreground(rgb)
    pix=rgb[m].astype(np.int16)
    if len(pix)==0:
        return np.full(18,128,dtype=np.uint8)

    R,G,B=pix[:,0],pix[:,1],pix[:,2]

    rels=(
        trunc_div2_signed(R-G),
        trunc_div2_signed(G-B),
        trunc_div2_signed(B-R),
    )

    out=[]

    # 3 relations x 4 robust signed summaries = 12 states around origin128.
    for d in rels:
        vals=[
            signed_quantile(d,1,4),
            signed_quantile(d,1,2),
            signed_quantile(d,3,4),
            int(d.astype(np.int64).sum()//len(d)),
        ]
        out.extend(chroma_encode(v) for v in vals)

    # Direction dominance magnitudes live in positive hemisphere 0..127.
    for d in rels:
        out.append(ratio127(int((d>0).sum()),len(d)))
        out.append(ratio127(int((d<0).sum()),len(d)))

    q=np.asarray(out,dtype=np.uint8)
    assert q.shape==(18,)
    return q


def observe(p):
    rgb=load_rgb(p)
    return shape_state(rgb),color_state(rgb)


def records(root,split,limit):
    rows=[]
    for cls,(slabel,clabel) in CLASSES.items():
        for p in paths(root,split,cls,limit):
            s,c=observe(p)
            rows.append({
                "class":cls,
                "shape_label":slabel,
                "color_label":clabel,
                "shape":s,
                "color":c,
                "path":str(p),
            })
    return rows


def nearest(q,memory):
    best=None
    for i,(z,label) in enumerate(memory):
        E=energy(q,z)
        key=(E,i)
        if best is None or key<best[0]:
            best=(key,label,E,i)
    _,label,E,i=best
    return label,E,i


def whole_nearest(s,c,memory):
    q=np.concatenate([s,c])
    best=None
    for i,(ss,cc,pair) in enumerate(memory):
        E=energy(q,np.concatenate([ss,cc]))
        key=(E,i)
        if best is None or key<best[0]:
            best=(key,pair,E,i)
    _,pair,E,i=best
    return pair,E,i


def memories(train,heldout_pair):
    sm=[];cm=[];wm=[]
    for r in train:
        pair=(r["shape_label"],r["color_label"])
        if pair==heldout_pair:
            continue
        sm.append((r["shape"],r["shape_label"]))
        cm.append((r["color"],r["color_label"]))
        wm.append((r["shape"],r["color"],pair))
    return sm,cm,wm


def evaluate_fold(train,test,heldout_class):
    truth=CLASSES[heldout_class]
    sm,cm,wm=memories(train,truth)

    assert any(lbl==truth[0] for _,lbl in sm)
    assert any(lbl==truth[1] for _,lbl in cm)
    assert all(pair!=truth for _,_,pair in wm)

    rows=[r for r in test if r["class"]==heldout_class]

    sc=cc=pc=wc=0
    shape_E=[];color_E=[];whole_E=[]

    for r in rows:
        ps,Es,_=nearest(r["shape"],sm)
        pcol,Ec,_=nearest(r["color"],cm)
        pair=(ps,pcol)

        wp,Ew,_=whole_nearest(r["shape"],r["color"],wm)

        sc+=int(ps==truth[0])
        cc+=int(pcol==truth[1])
        pc+=int(pair==truth)
        wc+=int(wp==truth)

        shape_E.append(Es)
        color_E.append(Ec)
        whole_E.append(Ew)

    return {
        "heldout_class":heldout_class,
        "heldout_pair":{"shape":truth[0],"color":truth[1]},
        "test_images":len(rows),
        "shape_correct":sc,
        "color_correct":cc,
        "constructive_exact_pair_correct":pc,
        "whole_pattern_exact_pair_correct":wc,
        "shape_measure_mean":float(np.mean(shape_E)),
        "color_measure_mean":float(np.mean(color_E)),
        "whole_measure_mean":float(np.mean(whole_E)),
    }


def balanced_orthogonality(train):
    by_class={}
    for cls in CLASSES:
        by_class[cls]=[r for r in train if r["class"]==cls][:50]

    queries=[r for cls in CLASSES for r in by_class[cls]]

    shape_ok=color_ok=0
    shape_m=[];color_m=[]

    for q in queries:
        ssdc=[
            r for r in train
            if r["shape_label"]==q["shape_label"]
            and r["color_label"]!=q["color_label"]
        ]
        dssc=[
            r for r in train
            if r["shape_label"]!=q["shape_label"]
            and r["color_label"]==q["color_label"]
        ]

        EsS=min(energy(q["shape"],r["shape"]) for r in ssdc)
        EdS=min(energy(q["shape"],r["shape"]) for r in dssc)

        EsC=min(energy(q["color"],r["color"]) for r in ssdc)
        EdC=min(energy(q["color"],r["color"]) for r in dssc)

        shape_ok+=int(EsS<EdS)
        color_ok+=int(EdC<EsC)
        shape_m.append(EdS-EsS)
        color_m.append(EsC-EdC)

    return {
        "queries":len(queries),
        "shape_prefers_same_shape_cross_color":[shape_ok,len(queries)],
        "color_prefers_same_color_cross_shape":[color_ok,len(queries)],
        "shape_margin_mean":float(np.mean(shape_m)),
        "color_margin_mean":float(np.mean(color_m)),
    }


def state_gates():
    # Shape never crosses the positive hemisphere.
    dummy=np.zeros((100,100,3),dtype=np.uint8)
    s=shape_state(dummy)
    assert int(s.max())<=127

    # Signed chroma polarity around origin 128.
    orange=np.zeros((10,10,3),dtype=np.uint8)
    orange[:]=[220,120,30]
    green=np.zeros((10,10,3),dtype=np.uint8)
    green[:]=[70,180,50]

    co=color_state(orange)
    cg=color_state(green)

    # First relation is R-G. Median is slot 1.
    assert int(co[1])<128
    assert int(cg[1])>128

    return {
        "pass":True,
        "shape_domain":[0,127],
        "color_signed_origin":128,
        "orange_RG_median_state":int(co[1]),
        "green_RG_median_state":int(cg[1]),
    }


def main(root,out):
    root=Path(root)

    train=records(root,"Training",MAX_TRAIN_PER_CLASS)
    test=records(root,"Test",MAX_TEST_PER_CLASS)

    gates=state_gates()
    orth=balanced_orthogonality(train)

    folds=[evaluate_fold(train,test,cls) for cls in CLASSES]

    total=sum(f["test_images"] for f in folds)
    shape=sum(f["shape_correct"] for f in folds)
    color=sum(f["color_correct"] for f in folds)
    pair=sum(f["constructive_exact_pair_correct"] for f in folds)
    whole=sum(f["whole_pattern_exact_pair_correct"] for f in folds)

    report={
        "model":"v21-hemisphere-aware-orthogonal-observers",
        "parent_v20":{
            "shape_correct":"235/480",
            "color_correct":"240/480",
            "constructive_exact":"124/480",
        },
        "mprc_corrections":{
            "shape":"positive magnitudes restricted to one 0..127 hemisphere",
            "color":"signed R-G/G-B/B-R relations encoded around origin 128",
            "distance":"Z256 circular cdist",
            "shape_structure":"16 row bands + 7 column bands + integer boundary geometry; no resize",
        },
        "state_gates":gates,
        "orthogonality":orth,
        "folds":folds,
        "aggregate":{
            "test_images":total,
            "shape_correct":shape,
            "color_correct":color,
            "constructive_exact_pair_correct":pair,
            "whole_pattern_exact_pair_correct":whole,
        },
        "claim_boundary":(
            "v21 tests whether MPRC-native factor encoding fixes v20's OBSERVE/IDENTIFY "
            "failure on a curated real-image shape/color cross. Factor labels define the "
            "resolved LUTs; this is not unsupervised factor discovery or the final 14,464-state "
            "Arshad-ViT manifold encoder."
        ),
    }

    p=Path(out)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print(json.dumps({
        "gates":gates,
        "orthogonality":orth,
        "aggregate":report["aggregate"],
        "folds":folds,
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",default="results/hemisphere_orthogonal_observers_v21.json")
    args=ap.parse_args()
    main(args.root,args.out)
