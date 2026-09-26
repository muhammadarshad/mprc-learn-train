"""v20 — Real orthogonal visual factors: SHAPE + COLOR -> constructed state.

Owner's example
===============
Orange and lime can look alike in shape while color distinguishes them.

Let:
    a1 = SHAPE
    a2 = COLOR

Real-data 2x2 factor cross from Fruits-360:
    round_citrus + orange -> Orange 1
    round_citrus + green  -> Limes 1
    pepper       + orange -> Pepper Orange 1
    pepper       + green  -> Pepper Green 1

Four folds
==========
In each fold, one WHOLE combination is removed from resolved object memory.

Example:
    hold out pepper+green entirely.

Resolved factor memory still contains:
    pepper shape  from Pepper Orange
    green color   from Limes

Therefore the constructive path must do:

    OBSERVE
      -> IDENTIFY shape state + color state
      -> MEASURE each against its own resolved factor LUT
      -> SELECT shape
      -> SELECT color
      -> BIND(selected_shape, selected_color)
      -> CONSTRUCT response

No whole-combination lookup participates in that path.

Holographic/full-pattern control
================================
A separate control stores concatenated [shape || color] states, but only for the
three SEEN combinations. It can only return a stored pair label.

Factor extraction
=================
All operations are integer.

SHAPE state (64 bytes):
    deterministic foreground mask on white-background Fruits-360 image
    32 row-band occupancies + 32 column-band occupancies, each mapped to uint8

COLOR state (12 bytes):
    integer R/G/B foreground statistics:
    q25, median, q75, integer mean for each channel

Core comparison:
    Z256 circular distance, wide integer MEASURE

This is not yet the final MPRC image manifold or INFORMATION codec. It is a
real-image factor-resolution gate designed to test the owner's exact orthogonal
feature argument without a conventional learned classifier.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

MAX_TRAIN_PER_CLASS=140
MAX_TEST_PER_CLASS=120

CLASSES={
    "Orange 1":        ("round_citrus","orange"),
    "Limes 1":         ("round_citrus","green"),
    "Pepper Orange 1": ("pepper","orange"),
    "Pepper Green 1":  ("pepper","green"),
}

SHAPES=("round_citrus","pepper")
COLORS=("orange","green")


def cdist(a,b):
    aa=np.asarray(a,dtype=np.int16)
    bb=np.asarray(b,dtype=np.int16)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def energy(a,b)->int:
    return int(cdist(a,b).sum(dtype=np.int64))


def image_paths(root:Path,split:str,cls:str,limit:int):
    d=root/split/cls
    if not d.exists():
        raise FileNotFoundError(d)
    xs=sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])
    return xs[:limit]


def load_rgb(p:Path)->np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8)


def foreground_mask(rgb:np.ndarray)->np.ndarray:
    """Dataset-observation boundary, not the MPRC relation itself.

    Fruits-360 uses a near-white background. Keep a pixel when at least one RGB
    channel is below 245. This is deterministic integer segmentation.
    """
    return np.any(rgb<245,axis=2)


def occupancy_bytes(mask:np.ndarray,axis:int,bands:int=32)->np.ndarray:
    """Band occupancy mapped exactly to 0..255 with integer arithmetic."""
    parts=np.array_split(np.arange(mask.shape[axis]),bands)
    out=[]

    for idx in parts:
        if axis==0:
            z=mask[idx,:]
        else:
            z=mask[:,idx]

        num=int(z.sum())
        den=int(z.size)
        out.append((num*255 + den//2)//den)

    return np.asarray(out,dtype=np.uint8)


def shape_state(rgb:np.ndarray)->np.ndarray:
    m=foreground_mask(rgb)
    row=occupancy_bytes(m,axis=0,bands=32)
    col=occupancy_bytes(m,axis=1,bands=32)
    out=np.concatenate([row,col])
    assert out.shape==(64,)
    return out


def kth(vals:np.ndarray,num:int,den:int)->int:
    """Integer order statistic at floor((n-1)*num/den)."""
    x=np.asarray(vals,dtype=np.uint8).reshape(-1)
    if len(x)==0:
        return 255
    k=((len(x)-1)*num)//den
    y=np.partition(x,k)
    return int(y[k])


def color_state(rgb:np.ndarray)->np.ndarray:
    m=foreground_mask(rgb)
    pix=rgb[m]

    if len(pix)==0:
        return np.full(12,255,dtype=np.uint8)

    out=[]
    for ch in range(3):
        v=pix[:,ch]
        out.extend([
            kth(v,1,4),
            kth(v,1,2),
            kth(v,3,4),
            int(v.astype(np.uint64).sum()//len(v)),
        ])

    z=np.asarray(out,dtype=np.uint8)
    assert z.shape==(12,)
    return z


def observe(p:Path):
    rgb=load_rgb(p)
    return shape_state(rgb),color_state(rgb)


def build_records(root:Path,split:str,limit:int):
    rows=[]
    for cls,(shape,color) in CLASSES.items():
        for p in image_paths(root,split,cls,limit):
            s,c=observe(p)
            rows.append({
                "class":cls,
                "shape_label":shape,
                "color_label":color,
                "shape":s,
                "color":c,
                "path":str(p),
            })
    return rows


def nearest_factor(query:np.ndarray,memory:list[tuple[np.ndarray,str]]):
    best=None

    for i,(state,label) in enumerate(memory):
        E=energy(query,state)
        key=(E,i)

        if best is None or key<best[0]:
            best=(key,label,E,i)

    _,label,E,i=best
    return label,E,i


def nearest_whole(shape:np.ndarray,color:np.ndarray,memory):
    """Whole-pattern memory control: can only return a remembered factor pair."""
    q=np.concatenate([shape,color])
    best=None

    for i,(s,c,pair) in enumerate(memory):
        z=np.concatenate([s,c])
        E=energy(q,z)
        key=(E,i)

        if best is None or key<best[0]:
            best=(key,pair,E,i)

    _,pair,E,i=best
    return pair,E,i


def factor_memory(train_rows,heldout_pair):
    shape_mem=[]
    color_mem=[]
    whole_mem=[]

    for r in train_rows:
        pair=(r["shape_label"],r["color_label"])

        if pair==heldout_pair:
            continue

        # Resolved factor states are reusable across combinations.
        shape_mem.append((r["shape"],r["shape_label"]))
        color_mem.append((r["color"],r["color_label"]))

        # Whole-object memory deliberately stores only seen combinations.
        whole_mem.append((r["shape"],r["color"],pair))

    return shape_mem,color_mem,whole_mem


def factor_availability(shape_mem,color_mem,heldout_pair):
    s,c=heldout_pair
    return (
        any(lbl==s for _,lbl in shape_mem),
        any(lbl==c for _,lbl in color_mem),
    )


def evaluate_fold(train_rows,test_rows,heldout_class):
    heldout_pair=CLASSES[heldout_class]
    smem,cmem,wmem=factor_memory(train_rows,heldout_pair)

    s_ok,c_ok=factor_availability(smem,cmem,heldout_pair)
    assert s_ok and c_ok

    # Verify whole object combination is genuinely absent.
    assert all(pair!=heldout_pair for _,_,pair in wmem)

    rows=[r for r in test_rows if r["class"]==heldout_class]
    assert rows

    construct_exact=0
    shape_correct=0
    color_correct=0
    whole_exact=0
    whole_shape=0
    whole_color=0

    shape_E=[]
    color_E=[]
    whole_E=[]

    examples=[]

    for r in rows:
        ps,Es,_=nearest_factor(r["shape"],smem)
        pc,Ec,_=nearest_factor(r["color"],cmem)

        # BIND two independently SELECTed factors.
        constructed=(ps,pc)

        wp,Ew,_=nearest_whole(r["shape"],r["color"],wmem)

        truth=heldout_pair

        shape_correct+=int(ps==truth[0])
        color_correct+=int(pc==truth[1])
        construct_exact+=int(constructed==truth)

        whole_exact+=int(wp==truth)  # must remain zero: truth label absent.
        whole_shape+=int(wp[0]==truth[0])
        whole_color+=int(wp[1]==truth[1])

        shape_E.append(Es)
        color_E.append(Ec)
        whole_E.append(Ew)

        if len(examples)<8:
            examples.append({
                "file":Path(r["path"]).name,
                "truth":{"shape":truth[0],"color":truth[1]},
                "constructed":{"shape":ps,"color":pc},
                "shape_measure":Es,
                "color_measure":Ec,
                "whole_memory_selected":{"shape":wp[0],"color":wp[1]},
                "whole_memory_measure":Ew,
            })

    return {
        "heldout_class":heldout_class,
        "heldout_pair":{"shape":heldout_pair[0],"color":heldout_pair[1]},
        "whole_pair_present_in_memory":False,
        "shape_factor_present_in_memory":s_ok,
        "color_factor_present_in_memory":c_ok,
        "test_images":len(rows),
        "constructive":{
            "shape_correct":shape_correct,
            "color_correct":color_correct,
            "exact_pair_correct":construct_exact,
            "shape_measure_mean":float(np.mean(shape_E)),
            "color_measure_mean":float(np.mean(color_E)),
        },
        "whole_pattern_memory_control":{
            "exact_pair_correct":whole_exact,
            "shape_factor_correct":whole_shape,
            "color_factor_correct":whole_color,
            "measure_mean":float(np.mean(whole_E)),
        },
        "examples":examples,
    }


def shape_color_orthogonality_diagnostic(train_rows):
    """Measure whether factor states behave as intended.

    For each observation, compare nearest:
      same-shape/different-color vs different-shape/same-color
    separately in SHAPE and COLOR spaces.

    Desired:
      shape space should prefer same shape despite color change;
      color space should prefer same color despite shape change.
    """
    ss=cc=total=0
    margins_shape=[]
    margins_color=[]

    # cap diagnostic cost
    query_rows=train_rows[:min(300,len(train_rows))]

    for q in query_rows:
        same_shape_diff_color=[
            r for r in train_rows
            if r["shape_label"]==q["shape_label"]
            and r["color_label"]!=q["color_label"]
        ]
        diff_shape_same_color=[
            r for r in train_rows
            if r["shape_label"]!=q["shape_label"]
            and r["color_label"]==q["color_label"]
        ]

        if not same_shape_diff_color or not diff_shape_same_color:
            continue

        Es_shape=min(energy(q["shape"],r["shape"]) for r in same_shape_diff_color)
        Ed_shape=min(energy(q["shape"],r["shape"]) for r in diff_shape_same_color)

        Es_color=min(energy(q["color"],r["color"]) for r in same_shape_diff_color)
        Ed_color=min(energy(q["color"],r["color"]) for r in diff_shape_same_color)

        # Shape factor should ignore color and stay nearer same shape.
        ss+=int(Es_shape<Ed_shape)

        # Color factor should ignore shape and stay nearer same color.
        cc+=int(Ed_color<Es_color)

        margins_shape.append(Ed_shape-Es_shape)
        margins_color.append(Es_color-Ed_color)
        total+=1

    return {
        "queries":total,
        "shape_prefers_same_shape_cross_color":[ss,total],
        "color_prefers_same_color_cross_shape":[cc,total],
        "shape_margin_mean":float(np.mean(margins_shape)) if margins_shape else 0.0,
        "color_margin_mean":float(np.mean(margins_color)) if margins_color else 0.0,
    }


def main(root:str,out_path:str):
    root=Path(root)

    train=build_records(root,"Training",MAX_TRAIN_PER_CLASS)
    test=build_records(root,"Test",MAX_TEST_PER_CLASS)

    assert len(train)>0 and len(test)>0

    orth=shape_color_orthogonality_diagnostic(train)

    folds=[]
    for cls in CLASSES:
        folds.append(evaluate_fold(train,test,cls))

    construct=sum(f["constructive"]["exact_pair_correct"] for f in folds)
    whole=sum(f["whole_pattern_memory_control"]["exact_pair_correct"] for f in folds)
    total=sum(f["test_images"] for f in folds)

    shape=sum(f["constructive"]["shape_correct"] for f in folds)
    color=sum(f["constructive"]["color_correct"] for f in folds)

    report={
        "model":"v20-real-shape-color-constructive-resolution",
        "dataset":{
            "name":"Fruits-360 100x100",
            "classes":CLASSES,
            "train_images_used":len(train),
            "test_images_used":len(test),
            "source_is_real_images":True,
            "factor_pairs_synthesized":False,
        },
        "operation":{
            "OBSERVE":"real RGB image",
            "IDENTIFY":"integer shape state (64 bytes) + color state (12 bytes)",
            "factor_memory":"resolved shape/color states from seen combinations",
            "MEASURE":"sum Z256 circular distances",
            "SELECT":"nearest resolved factor state independently",
            "BIND":"(selected_shape, selected_color)",
            "CONSTRUCT":"factor pair response even when whole pair absent from memory",
        },
        "orthogonality_diagnostic":orth,
        "folds":folds,
        "aggregate":{
            "test_images":total,
            "shape_correct":shape,
            "color_correct":color,
            "constructive_exact_pair_correct":construct,
            "whole_pattern_memory_exact_pair_correct":whole,
        },
        "claim_boundary":(
            "v20 uses real Fruits-360 images and a curated 2x2 shape/color factor cross. "
            "The foreground observation extractor is deterministic integer preprocessing, "
            "not yet the final Arshad-ViT manifold/INFORMATION encoder. Success demonstrates "
            "independent factor resolution and construction of an omitted whole combination; "
            "it does not yet establish arbitrary natural-scene factor discovery."
        ),
    }

    out=Path(out_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print(json.dumps({
        "orthogonality":orth,
        "aggregate":report["aggregate"],
        "fold_summary":[
            {
                "heldout":f["heldout_class"],
                "constructive":[f["constructive"]["exact_pair_correct"],f["test_images"]],
                "whole_memory":[f["whole_pattern_memory_control"]["exact_pair_correct"],f["test_images"]],
            }
            for f in folds
        ]
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--out",default="results/real_shape_color_constructive_v20.json")
    args=ap.parse_args()
    main(args.root,args.out)
