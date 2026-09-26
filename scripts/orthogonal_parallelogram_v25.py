"""v25 — Orthogonal factor parallelogram construction.

Main-agenda gate:
    OBSERVE(16ch)
      -> structural transpose
      -> resolved factor displacement
      -> BIND/MOVE construct missing corner
      -> REACT
      -> MEASURE
      -> SELECT

No handcrafted shape/color vectors and no channel selection.

Conditional theorem:
    If X[s,c] = B + S[s] + C[c] in the Z256 module, then
        X[a,b] = X[a,1-b] + X[1-a,b] - X[1-a,1-b]  (mod 256).

Real-data question:
    Does the constructed held-out wave state beat all three remembered whole
    states on real Orange/Lime/Pepper queries?
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
    "Orange 1":        (0, 0, "citrus", "orange"),
    "Limes 1":         (0, 1, "citrus", "green"),
    "Pepper Orange 1": (1, 0, "pepper", "orange"),
    "Pepper Green 1":  (1, 1, "pepper", "green"),
}

PAIR_TO_CLASS = {
    (a,b): cls
    for cls,(a,b,_,_) in CLASSES.items()
}


def zadd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.uint16) + b.astype(np.uint16)) & 0xFF).astype(np.uint8)


def zsub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a.astype(np.int16) - b.astype(np.int16)) & 0xFF).astype(np.uint8)


def ring_measure(a: np.ndarray, b: np.ndarray) -> int:
    aa = a.astype(np.int16, copy=False)
    bb = b.astype(np.int16, copy=False)
    ab = (aa - bb) & 0xFF
    ba = (bb - aa) & 0xFF
    d = np.minimum(ab, ba)
    return int(d.sum(dtype=np.int64))


def ring_measure_per_channel(a: np.ndarray, b: np.ndarray) -> list[int]:
    assert a.shape == b.shape == (16,128,113)
    return [ring_measure(a[k], b[k]) for k in range(16)]


def frame_rgb_lossless(path: Path) -> np.ndarray:
    """Center a source image in the fixed 113x128 frame without resizing.

    Fruits-360 test images are 100x100, so every source byte is preserved.
    """
    im = Image.open(path).convert("RGB")
    if im.width > FW or im.height > FH:
        raise ValueError(
            f"{path} is {im.width}x{im.height}, exceeds fixed frame {FW}x{FH}"
        )

    canvas = Image.new("RGB", (FW, FH), (255,255,255))
    ax = (FW - im.width) // 2
    ay = (FH - im.height) // 2
    canvas.paste(im, (ax, ay))
    arr = np.asarray(canvas, dtype=np.uint8)
    assert arr.shape == (FH, FW, 3)
    return arr


def encode_paths(paths: list[Path], encoder: QCMWaveEncoder) -> list[np.ndarray]:
    states = []
    for p in paths:
        rgb = frame_rgb_lossless(p)
        x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()
        with torch.no_grad():
            psi = encoder(x)
        assert float(torch.abs(psi - psi.floor()).max().item()) == 0.0
        z = psi[0].to(torch.uint8).cpu().numpy()       # [16,113,128]

        # v24 candidate coordinate interface: preserve channels, transpose frame.
        structural = np.transpose(z, (0,2,1)).copy()  # [16,128,113]
        assert structural.shape == (16,128,113)
        states.append(structural)
    return states


def list_images(root: Path, split: str, cls: str, limit: int) -> list[Path]:
    d = root / split / cls
    xs = sorted(
        p for p in d.iterdir()
        if p.suffix.lower() in {".jpg",".jpeg",".png"}
    )
    return xs[:limit]


def build_records(root: Path, split: str, limit: int, encoder):
    rows = []
    for cls,(a,b,shape,color) in CLASSES.items():
        paths = list_images(root, split, cls, limit)
        states = encode_paths(paths, encoder)
        for p,state in zip(paths, states):
            rows.append({
                "class": cls,
                "pair": (a,b),
                "shape": shape,
                "color": color,
                "path": p,
                "state": state,
            })
    return rows


def class_records(rows, cls):
    return [r for r in rows if r["class"] == cls]


def medoid(records):
    """Resolved representative using full 16-channel ring MEASURE."""
    n = len(records)
    totals = []

    for i,a in enumerate(records):
        s = 0
        for j,b in enumerate(records):
            if i == j:
                continue
            s += ring_measure(a["state"], b["state"])
        totals.append((s, i))

    totals.sort()
    total, idx = totals[0]
    r = records[idx]

    return {
        "class": r["class"],
        "pair": r["pair"],
        "shape": r["shape"],
        "color": r["color"],
        "path": r["path"],
        "state": r["state"],
        "within_class_total_energy": int(total),
    }


def construct_missing(medoids: dict[str,dict], heldout_class: str):
    a,b,shape,color = CLASSES[heldout_class]

    same_shape_other_color = PAIR_TO_CLASS[(a,1-b)]
    other_shape_same_color = PAIR_TO_CLASS[(1-a,b)]
    opposite = PAIR_TO_CLASS[(1-a,1-b)]

    A = medoids[same_shape_other_color]["state"]
    B = medoids[other_shape_same_color]["state"]
    C = medoids[opposite]["state"]

    # X_ab = X_a,1-b + X_1-a,b - X_1-a,1-b
    constructed = zsub(zadd(A,B), C)

    return {
        "state": constructed,
        "sources": {
            "same_shape_other_color": same_shape_other_color,
            "other_shape_same_color": other_shape_same_color,
            "opposite": opposite,
        },
        "target_pair": (a,b),
        "target_shape": shape,
        "target_color": color,
    }


def attention_bank_energy(candidate: np.ndarray, query: np.ndarray):
    assert candidate.shape == query.shape == (16,128,113)
    total = 0
    per = []

    for k in range(16):
        out = attention_forward(
            candidate[k],
            query[k],
            lut=None,
            rounds=ROUNDS,
            return_physical_state=False,
        )
        e = int(out["energy"])
        per.append(e)
        total += e

    return total, per


def evaluate_fold(train, test, heldout_class):
    # Build medoids for all four classes for diagnostics, but the held-out medoid
    # is NEVER available to construction or SELECT.
    medoids = {
        cls: medoid(class_records(train, cls))
        for cls in CLASSES
    }

    visible_classes = [c for c in CLASSES if c != heldout_class]
    visible = {c:medoids[c] for c in visible_classes}

    constructed = construct_missing(visible, heldout_class)
    qrows = class_records(test, heldout_class)

    selected_constructed_raw = 0
    selected_constructed_attention = 0
    raw_margins = []
    attention_margins = []
    rows = []

    hidden_medoid = medoids[heldout_class]

    # Diagnostic: how close is constructed state to the hidden resolved target medoid?
    construction_to_hidden_raw = ring_measure(
        constructed["state"], hidden_medoid["state"]
    )

    for q in qrows:
        candidates = {
            cls: visible[cls]["state"]
            for cls in visible_classes
        }
        candidates["CONSTRUCTED"] = constructed["state"]

        raw_scores = {
            name: ring_measure(state, q["state"])
            for name,state in candidates.items()
        }

        attn_scores = {}
        attn_per_channel = {}
        for name,state in candidates.items():
            e, per = attention_bank_energy(state, q["state"])
            attn_scores[name] = int(e)
            attn_per_channel[name] = [int(x) for x in per]

        best_visible_raw = min(raw_scores[c] for c in visible_classes)
        best_visible_attn = min(attn_scores[c] for c in visible_classes)

        raw_margin = int(best_visible_raw - raw_scores["CONSTRUCTED"])
        attn_margin = int(best_visible_attn - attn_scores["CONSTRUCTED"])

        raw_margins.append(raw_margin)
        attention_margins.append(attn_margin)

        raw_select = min(raw_scores, key=lambda n:(raw_scores[n], n))
        attn_select = min(attn_scores, key=lambda n:(attn_scores[n], n))

        selected_constructed_raw += int(raw_select == "CONSTRUCTED")
        selected_constructed_attention += int(attn_select == "CONSTRUCTED")

        rows.append({
            "query_file": q["path"].name,
            "raw_scores": {k:int(v) for k,v in raw_scores.items()},
            "attention_scores": {k:int(v) for k,v in attn_scores.items()},
            "raw_select": raw_select,
            "attention_select": attn_select,
            "raw_constructed_margin": raw_margin,
            "attention_constructed_margin": attn_margin,
            "constructed_attention_per_channel": attn_per_channel["CONSTRUCTED"],
        })

    return {
        "heldout_class": heldout_class,
        "target": {
            "pair": list(CLASSES[heldout_class][:2]),
            "shape": CLASSES[heldout_class][2],
            "color": CLASSES[heldout_class][3],
        },
        "visible_classes": visible_classes,
        "construction_sources": constructed["sources"],
        "constructed_vs_hidden_medoid_raw_energy": int(construction_to_hidden_raw),
        "queries": len(qrows),
        "raw_constructed_selected": int(selected_constructed_raw),
        "attention_constructed_selected": int(selected_constructed_attention),
        "raw_margin_mean": float(np.mean(raw_margins)),
        "raw_margin_median": float(np.median(raw_margins)),
        "attention_margin_mean": float(np.mean(attention_margins)),
        "attention_margin_median": float(np.median(attention_margins)),
        "rows": rows,
        "_constructed": constructed,
        "_medoids": medoids,
    }


def bank_to_rgb(bank: np.ndarray) -> Image.Image:
    """Visualize constructed state's RGB pass-through channels."""
    assert bank.shape == (16,128,113)
    psi = np.transpose(bank, (0,2,1))  # [16,113,128]
    rgb = np.transpose(psi[:3], (1,2,0)).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def thumb(im: Image.Image, size=(145,130)):
    x = im.copy().convert("RGB")
    x.thumbnail(size)
    c = Image.new("RGB", size, "white")
    c.paste(x, ((size[0]-x.width)//2,(size[1]-x.height)//2))
    return c


def photo_from_path(path: Path):
    return Image.open(path).convert("RGB")


def make_visual(folds, out_path: str):
    W = 1180
    row_h = 205
    H = 70 + len(folds)*row_h
    sheet = Image.new("RGB",(W,H),"white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    d.text((18,14),"v25 — 16-channel orthogonal parallelogram construction",fill="black",font=font)
    d.text((18,34),"three resolved medoids -> constructed missing RGB state -> held-out query example",fill="black",font=font)

    for i,f in enumerate(folds):
        y = 70+i*row_h
        med=f["_medoids"]
        con=f["_constructed"]

        source_names=[
            con["sources"]["same_shape_other_color"],
            con["sources"]["other_shape_same_color"],
            con["sources"]["opposite"],
        ]
        query_row=f["rows"][0]
        heldout=f["heldout_class"]

        # Find matching first test query path later from file name is unavailable here;
        # display hidden held-out medoid as diagnostic reference, explicitly labelled.
        panels=[
            (source_names[0], photo_from_path(med[source_names[0]]["path"])),
            (source_names[1], photo_from_path(med[source_names[1]]["path"])),
            ("subtract "+source_names[2], photo_from_path(med[source_names[2]]["path"])),
            ("CONSTRUCTED RGB", bank_to_rgb(con["state"])),
            ("hidden target medoid*", photo_from_path(med[heldout]["path"])),
        ]
        xs=[15,175,335,495,655]

        border=(35,130,55) if f["attention_constructed_selected"]>0 else (170,55,45)
        d.rectangle((8,y,1170,y+row_h-8),outline=border,width=3)

        for x,(label,im) in zip(xs,panels):
            d.text((x,y+8),label,fill="black",font=font)
            sheet.paste(thumb(im),(x,y+30))

        tx=820
        d.text((tx,y+8),f"held out: {heldout}",fill="black",font=font)
        d.text((tx,y+28),f"raw select construction: {f['raw_constructed_selected']}/{f['queries']}",fill="black",font=font)
        d.text((tx,y+48),f"attention select construction: {f['attention_constructed_selected']}/{f['queries']}",fill="black",font=font)
        d.text((tx,y+68),f"raw margin mean: {f['raw_margin_mean']:.1f}",fill="black",font=font)
        d.text((tx,y+88),f"attention margin mean: {f['attention_margin_mean']:.1f}",fill="black",font=font)
        d.text((tx,y+112),f"constructed -> hidden medoid raw E:",fill="black",font=font)
        d.text((tx,y+130),str(f["constructed_vs_hidden_medoid_raw_energy"]),fill="black",font=font)
        d.text((tx,y+160),"*hidden medoid is diagnostic only;",fill="black",font=font)
        d.text((tx,y+177),"not used by construction or SELECT",fill="black",font=font)

    sheet.save(out_path)


def strip_private(f):
    return {k:v for k,v in f.items() if not k.startswith("_")}


def main(root: str, out_json: str, out_png: str):
    root=Path(root)
    encoder=QCMWaveEncoder().eval()

    train=build_records(root,"Training",TRAIN_PER_CLASS,encoder)
    test=build_records(root,"Test",TEST_PER_CLASS,encoder)

    folds=[evaluate_fold(train,test,cls) for cls in CLASSES]

    raw_selected=sum(f["raw_constructed_selected"] for f in folds)
    attn_selected=sum(f["attention_constructed_selected"] for f in folds)
    total=sum(f["queries"] for f in folds)

    report={
        "model":"v25-orthogonal-parallelogram-construction",
        "algebra":{
            "candidate_factor_model":"X_sc = B + S_s + C_c mod 256",
            "completion":"X_ab = X_a,1-b + X_1-a,b - X_1-a,1-b mod 256",
            "status":"exact theorem conditional on additive orthogonal-factor model",
        },
        "observation":{
            "encoder":"owner QCMWaveEncoder",
            "channels":16,
            "source_frame":[113,128],
            "structural_frame":[128,113],
            "source_bytes_preserved_without_resize":True,
            "manual_channel_selection":False,
        },
        "attention":{
            "path":"TRANSPORT -> BIND -> REACT -> MEASURE",
            "rounds":ROUNDS,
            "lut":"identity",
            "all_channels_used":True,
        },
        "folds":[strip_private(f) for f in folds],
        "aggregate":{
            "queries":int(total),
            "raw_constructed_selected":int(raw_selected),
            "attention_constructed_selected":int(attn_selected),
        },
        "claim_boundary":(
            "v25 directly tests whether real 16-channel wave observations are "
            "approximately separable by additive orthogonal factor displacement. "
            "The parallelogram law is exact only under that factor model. Real-data "
            "success/failure is empirical and does not alter the frozen ring algebra."
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
            "raw_constructed":[f["raw_constructed_selected"],f["queries"]],
            "attention_constructed":[f["attention_constructed_selected"],f["queries"]],
            "raw_margin_mean":f["raw_margin_mean"],
            "attention_margin_mean":f["attention_margin_mean"],
            "constructed_vs_hidden_medoid_raw_energy":f["constructed_vs_hidden_medoid_raw_energy"],
        } for f in folds]
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--json",default="results/v25_orthogonal_parallelogram.json")
    ap.add_argument("--png",default="results/v25_orthogonal_parallelogram.png")
    a=ap.parse_args()
    main(a.root,a.json,a.png)
