"""v26 — Relational IDENTIFY over the 16-channel MPRC wave state.

Main agenda:
    OBSERVE
      -> IDENTIFY factor evidence from visible relations
      -> independent factor SELECT
      -> construct absent pair identity
      -> compare raw vs frozen BIND->REACT->MEASURE evidence

No handcrafted shape/color feature vectors.
No manual channel semantics.
No top-K channel selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from wave_encoder_torch import QCMWaveEncoder, FH, FW
from src.mprc_structural.attention import attention_forward

TRAIN_PER_CLASS = 8
TEST_PER_CLASS = 8
ROUNDS = 7

CLASSES = {
    "Orange 1":        ("citrus", "orange"),
    "Limes 1":         ("citrus", "green"),
    "Pepper Orange 1": ("pepper", "orange"),
    "Pepper Green 1":  ("pepper", "green"),
}

CHANNELS = [
    "R","G","B","Luma","Gray","Chroma","Phase","Winding",
    "d1(R-G)","d2(G-B)",
    "LoG@0","LoG@1","LoG@4","LoG@16","LoG@64","LoG@128",
]


def ring_measure(a: np.ndarray, b: np.ndarray) -> int:
    aa=a.astype(np.int16,copy=False)
    bb=b.astype(np.int16,copy=False)
    ab=(aa-bb)&0xFF
    ba=(bb-aa)&0xFF
    d=np.minimum(ab,ba)
    return int(d.sum(dtype=np.int64))


def per_channel_raw(a: np.ndarray, b: np.ndarray) -> list[int]:
    assert a.shape==b.shape==(16,128,113)
    return [ring_measure(a[k],b[k]) for k in range(16)]


def frame_rgb_lossless(path: Path) -> np.ndarray:
    im=Image.open(path).convert("RGB")
    if im.width>FW or im.height>FH:
        raise ValueError(f"{path} exceeds fixed frame")
    canvas=Image.new("RGB",(FW,FH),(255,255,255))
    ax=(FW-im.width)//2
    ay=(FH-im.height)//2
    canvas.paste(im,(ax,ay))
    arr=np.asarray(canvas,dtype=np.uint8)
    assert arr.shape==(FH,FW,3)
    return arr


def encode_one(path: Path, enc: QCMWaveEncoder) -> np.ndarray:
    rgb=frame_rgb_lossless(path)
    x=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()
    with torch.no_grad():
        y=enc(x)
    assert float(torch.abs(y-y.floor()).max().item())==0.0
    z=y[0].to(torch.uint8).cpu().numpy()          # [16,113,128]
    s=np.transpose(z,(0,2,1)).copy()              # [16,128,113]
    assert s.shape==(16,128,113)
    return s


def list_images(root:Path,split:str,cls:str,limit:int):
    d=root/split/cls
    return sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png"}
    )[:limit]


def build(root:Path,split:str,limit:int,enc):
    out=[]
    for cls,(shape,color) in CLASSES.items():
        for p in list_images(root,split,cls,limit):
            out.append({
                "class":cls,
                "shape":shape,
                "color":color,
                "path":p,
                "state":encode_one(p,enc),
            })
    return out


def records(rows,cls):
    return [r for r in rows if r["class"]==cls]


def medoid(rows):
    scores=[]
    for i,a in enumerate(rows):
        total=0
        for j,b in enumerate(rows):
            if i==j: continue
            total += ring_measure(a["state"],b["state"])
        scores.append((total,i))
    scores.sort()
    _,idx=scores[0]
    return rows[idx]


def resolved_medoids(train):
    return {cls:medoid(records(train,cls)) for cls in CLASSES}


def visible_relation_pairs(visible_classes):
    same_shape=[]
    same_color=[]
    for i,a in enumerate(visible_classes):
        for b in visible_classes[i+1:]:
            sa,ca=CLASSES[a]
            sb,cb=CLASSES[b]
            if sa==sb and ca!=cb:
                same_shape.append((a,b))
            if ca==cb and sa!=sb:
                same_color.append((a,b))
    assert len(same_shape)==1, same_shape
    assert len(same_color)==1, same_color
    return same_shape[0],same_color[0]


def discover_factor_channels(medoids,visible_classes):
    sp,cp=visible_relation_pairs(visible_classes)

    Ds=np.asarray(
        per_channel_raw(medoids[sp[0]]["state"],medoids[sp[1]]["state"]),
        dtype=np.int64
    )
    Dc=np.asarray(
        per_channel_raw(medoids[cp[0]]["state"],medoids[cp[1]]["state"]),
        dtype=np.int64
    )

    shape=[int(k) for k in np.where(Ds<Dc)[0]]
    color=[int(k) for k in np.where(Dc<Ds)[0]]
    shared=[int(k) for k in np.where(Dc==Ds)[0]]

    # An empty factor set would mean IDENTIFY has no evidence for that factor.
    return {
        "same_shape_pair":sp,
        "same_color_pair":cp,
        "D_same_shape":Ds,
        "D_same_color":Dc,
        "shape_channels":shape,
        "color_channels":color,
        "shared_channels":shared,
    }


def attention_vector(memory_state,query_state):
    vals=[]
    for k in range(16):
        out=attention_forward(
            memory_state[k],
            query_state[k],
            lut=None,
            rounds=ROUNDS,
            return_physical_state=False,
        )
        vals.append(int(out["energy"]))
    return vals


def sum_mask(vec,mask):
    if not mask:
        return None
    return int(sum(vec[k] for k in mask))


def factor_scores_from_medoids(query,visible_medoids,shape_channels,color_channels,use_attention):
    cache={}
    for cls,m in visible_medoids.items():
        if use_attention:
            cache[cls]=attention_vector(m["state"],query["state"])
        else:
            cache[cls]=per_channel_raw(m["state"],query["state"])

    shape_values=sorted(set(shape for shape,_ in CLASSES.values()))
    color_values=sorted(set(color for _,color in CLASSES.values()))

    shape_scores={}
    for s in shape_values:
        vals=[]
        for cls,m in visible_medoids.items():
            if m["shape"]==s:
                e=sum_mask(cache[cls],shape_channels)
                if e is not None:
                    vals.append(e)
        shape_scores[s]=min(vals) if vals else None

    color_scores={}
    for c in color_values:
        vals=[]
        for cls,m in visible_medoids.items():
            if m["color"]==c:
                e=sum_mask(cache[cls],color_channels)
                if e is not None:
                    vals.append(e)
        color_scores[c]=min(vals) if vals else None

    return shape_scores,color_scores,cache


def select_min(scores):
    valid=[(v,k) for k,v in scores.items() if v is not None]
    if not valid:
        return None
    valid.sort()
    return valid[0][1]


def whole_memory_control(query,visible_medoids,use_attention):
    scores={}
    for cls,m in visible_medoids.items():
        vec=attention_vector(m["state"],query["state"]) if use_attention else per_channel_raw(m["state"],query["state"])
        scores[cls]=int(sum(vec))
    selected=min(scores,key=lambda c:(scores[c],c))
    return selected,scores


def evaluate_fold(train,test,heldout):
    medoids=resolved_medoids(train)
    visible_classes=[c for c in CLASSES if c!=heldout]
    visible={c:medoids[c] for c in visible_classes}
    disc=discover_factor_channels(medoids,visible_classes)

    target_shape,target_color=CLASSES[heldout]
    rows=records(test,heldout)

    raw_shape=raw_color=raw_pair=0
    att_shape=att_color=att_pair=0
    out_rows=[]

    for q in rows:
        rs,rc,raw_cache=factor_scores_from_medoids(
            q,visible,
            disc["shape_channels"],disc["color_channels"],
            use_attention=False,
        )
        aps,apc,att_cache=factor_scores_from_medoids(
            q,visible,
            disc["shape_channels"],disc["color_channels"],
            use_attention=True,
        )

        p_rs=select_min(rs)
        p_rc=select_min(rc)
        p_as=select_min(aps)
        p_ac=select_min(apc)

        raw_construct=(p_rs,p_rc)
        att_construct=(p_as,p_ac)

        raw_shape += int(p_rs==target_shape)
        raw_color += int(p_rc==target_color)
        raw_pair += int(raw_construct==(target_shape,target_color))

        att_shape += int(p_as==target_shape)
        att_color += int(p_ac==target_color)
        att_pair += int(att_construct==(target_shape,target_color))

        raw_whole,raw_whole_scores=whole_memory_control(q,visible,False)
        att_whole,att_whole_scores=whole_memory_control(q,visible,True)

        out_rows.append({
            "query_file":q["path"].name,
            "raw_shape_scores":rs,
            "raw_color_scores":rc,
            "raw_constructed_pair":list(raw_construct),
            "attention_shape_scores":aps,
            "attention_color_scores":apc,
            "attention_constructed_pair":list(att_construct),
            "raw_whole_memory_select":raw_whole,
            "attention_whole_memory_select":att_whole,
            "raw_whole_scores":raw_whole_scores,
            "attention_whole_scores":att_whole_scores,
        })

    return {
        "heldout_class":heldout,
        "target":{"shape":target_shape,"color":target_color},
        "queries":len(rows),
        "relations":{
            "same_shape_different_color":list(disc["same_shape_pair"]),
            "same_color_different_shape":list(disc["same_color_pair"]),
        },
        "shape_channels":[{"index":k,"name":CHANNELS[k]} for k in disc["shape_channels"]],
        "color_channels":[{"index":k,"name":CHANNELS[k]} for k in disc["color_channels"]],
        "shared_channels":[{"index":k,"name":CHANNELS[k]} for k in disc["shared_channels"]],
        "channel_relation_values":[{
            "index":k,
            "name":CHANNELS[k],
            "same_shape_energy":int(disc["D_same_shape"][k]),
            "same_color_energy":int(disc["D_same_color"][k]),
            "role":(
                "shape" if k in disc["shape_channels"]
                else "color" if k in disc["color_channels"]
                else "shared"
            )
        } for k in range(16)],
        "raw":{
            "shape_correct":raw_shape,
            "color_correct":raw_color,
            "constructed_pair_correct":raw_pair,
        },
        "attention":{
            "shape_correct":att_shape,
            "color_correct":att_color,
            "constructed_pair_correct":att_pair,
        },
        "rows":out_rows,
        "_medoids":medoids,
    }


def make_visual(folds,out_path):
    W=1200
    row_h=220
    H=70+row_h*len(folds)
    sheet=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(sheet)
    font=ImageFont.load_default()

    d.text((15,12),"v26 Relational IDENTIFY — factor construction, not image synthesis",fill="black",font=font)
    d.text((15,32),"channel roles are discovered from visible same-shape and same-color relations",fill="black",font=font)

    for i,f in enumerate(folds):
        y=70+i*row_h
        med=f["_medoids"]
        target=f["heldout_class"]

        visible=[c for c in CLASSES if c!=target]
        x=15
        for cls in visible:
            im=Image.open(med[cls]["path"]).convert("RGB")
            im.thumbnail((130,130))
            canvas=Image.new("RGB",(130,130),"white")
            canvas.paste(im,((130-im.width)//2,(130-im.height)//2))
            sheet.paste(canvas,(x,y+32))
            d.text((x,y+12),cls,fill="black",font=font)
            x+=145

        hidden=Image.open(med[target]["path"]).convert("RGB")
        hidden.thumbnail((130,130))
        canvas=Image.new("RGB",(130,130),"white")
        canvas.paste(hidden,((130-hidden.width)//2,(130-hidden.height)//2))
        sheet.paste(canvas,(455,y+32))
        d.text((455,y+12),"hidden target*",fill="black",font=font)

        shape_names=",".join(x["name"] for x in f["shape_channels"]) or "NONE"
        color_names=",".join(x["name"] for x in f["color_channels"]) or "NONE"

        d.text((610,y+10),f"held out: {target}",fill="black",font=font)
        d.text((610,y+32),f"shape channels: {shape_names}",fill="black",font=font)
        d.text((610,y+54),f"color channels: {color_names}",fill="black",font=font)

        d.text((610,y+84),f"RAW pair: {f['raw']['constructed_pair_correct']}/{f['queries']}",fill="black",font=font)
        d.text((610,y+106),f"ATTN pair: {f['attention']['constructed_pair_correct']}/{f['queries']}",fill="black",font=font)
        d.text((610,y+132),f"RAW shape/color: {f['raw']['shape_correct']}/{f['queries']}  {f['raw']['color_correct']}/{f['queries']}",fill="black",font=font)
        d.text((610,y+154),f"ATTN shape/color: {f['attention']['shape_correct']}/{f['queries']}  {f['attention']['color_correct']}/{f['queries']}",fill="black",font=font)
        d.text((610,y+182),"*diagnostic only; hidden whole pair absent from memory",fill="black",font=font)

    sheet.save(out_path)


def clean_fold(f):
    return {k:v for k,v in f.items() if not k.startswith("_")}


def main(root,out_json,out_png):
    root=Path(root)
    enc=QCMWaveEncoder().eval()
    train=build(root,"Training",TRAIN_PER_CLASS,enc)
    test=build(root,"Test",TEST_PER_CLASS,enc)

    folds=[evaluate_fold(train,test,cls) for cls in CLASSES]

    total=sum(f["queries"] for f in folds)

    report={
        "model":"v26-relational-identify",
        "observation":{
            "channels":16,
            "manual_channel_semantics":False,
            "top_k":False,
            "source_resize":False,
        },
        "identify":{
            "shape_rule":"D(same-shape,diff-color) < D(same-color,diff-shape)",
            "color_rule":"D(same-color,diff-shape) < D(same-shape,diff-color)",
            "shared_rule":"equal energies => undecided/shared",
        },
        "folds":[clean_fold(f) for f in folds],
        "aggregate":{
            "queries":total,
            "raw_shape_correct":sum(f["raw"]["shape_correct"] for f in folds),
            "raw_color_correct":sum(f["raw"]["color_correct"] for f in folds),
            "raw_constructed_pair_correct":sum(f["raw"]["constructed_pair_correct"] for f in folds),
            "attention_shape_correct":sum(f["attention"]["shape_correct"] for f in folds),
            "attention_color_correct":sum(f["attention"]["color_correct"] for f in folds),
            "attention_constructed_pair_correct":sum(f["attention"]["constructed_pair_correct"] for f in folds),
        },
        "claim_boundary":(
            "v26 is a controlled factor-composition test. Factor labels define the "
            "two-axis benchmark, while channel roles are derived only from visible "
            "relations. It does not establish unsupervised arbitrary factor discovery."
        ),
    }

    p=Path(out_json)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(report,indent=2),encoding="utf-8")
    make_visual(folds,out_png)

    print(json.dumps({
        "aggregate":report["aggregate"],
        "folds":[{
            "heldout":f["heldout_class"],
            "shape_channels":[x["name"] for x in f["shape_channels"]],
            "color_channels":[x["name"] for x in f["color_channels"]],
            "raw":f["raw"],
            "attention":f["attention"],
        } for f in folds]
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--json",default="results/v26_relational_identify.json")
    ap.add_argument("--png",default="results/v26_relational_identify.png")
    a=ap.parse_args()
    main(a.root,a.json,a.png)
