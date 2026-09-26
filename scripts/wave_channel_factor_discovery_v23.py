"""v23 — 16-channel QCM wave encoder: orthogonal factor discovery.

Uses the owner's exact QCMWaveEncoder as OBSERVE.

Real 2x2 factor cross:
    citrus + orange -> Orange 1
    citrus + green  -> Limes 1
    pepper + orange -> Pepper Orange 1
    pepper + green  -> Pepper Green 1

For each held-out whole combination, only the other three combinations are
available to resolved memory.

Those three visible combinations contain:
  * one pair with SAME SHAPE, DIFFERENT COLOR
  * one pair with SAME COLOR, DIFFERENT SHAPE

For each wave channel k, derive from training data only:

    D_shape[k] = ring MEASURE(same-shape, different-color)
    D_color[k] = ring MEASURE(same-color, different-shape)

    score[k] = (D_color[k] - D_shape[k]) /
               (D_color[k] + D_shape[k] + 1)

Interpretation:
    score > 0 : channel preserves shape more strongly across color change
    score < 0 : channel preserves color more strongly across shape change

No channel is manually declared shape or color.

Top-K positive/negative channels are then used independently to IDENTIFY
shape and color for the omitted combination, BIND the two selected factors,
and construct the response.

The run also exports actual-photo audit sheets:
  QUERY | nearest SHAPE memory | nearest COLOR memory | nearest ALL-16 memory
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from wave_encoder_torch import QCMWaveEncoder, FH, FW

TRAIN_PER_CLASS = 16
TEST_PER_CLASS = 24
BATCH = 8
TOP_K = 4

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


def ring_distance_u8(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a.astype(np.int16, copy=False)
    bb = b.astype(np.int16, copy=False)
    d = np.abs(aa - bb)
    return np.minimum(d, 256 - d)


def frame_rgb(path: Path) -> np.ndarray:
    """Aspect-fit RGB into the encoder's fixed 113x128 frame.

    Framing is outside QCMWaveEncoder. Source pixels remain uint8.
    White background matches the Fruits-360 source background.
    """
    im = Image.open(path).convert("RGB")
    scale = min(FW / im.width, FH / im.height)
    nw = max(1, int(round(im.width * scale)))
    nh = max(1, int(round(im.height * scale)))
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (FW, FH), (255,255,255))
    ax = (FW - nw) // 2
    ay = (FH - nh) // 2
    canvas.paste(im, (ax, ay))
    arr = np.asarray(canvas, dtype=np.uint8)
    assert arr.shape == (FH, FW, 3)
    return arr


def list_images(root: Path, split: str, cls: str, limit: int):
    d = root / split / cls
    xs = sorted(p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    return xs[:limit]


def encode_paths(paths, encoder):
    out = []
    integer_ok = True

    for i in range(0, len(paths), BATCH):
        chunk = paths[i:i+BATCH]
        arr = np.stack([frame_rgb(p) for p in chunk], axis=0)
        x = torch.from_numpy(arr).permute(0,3,1,2).float()

        with torch.no_grad():
            y = encoder(x)

        fractional = torch.abs(y - y.floor()).max().item()
        integer_ok = integer_ok and (fractional == 0.0)

        z = y.to(torch.uint8).cpu().numpy()
        assert z.shape[1:] == (16, FH, FW)
        out.extend(list(z))

    return out, integer_ok


def build_records(root: Path, split: str, limit: int, encoder):
    paths = []
    meta = []

    for cls, (shape, color) in CLASSES.items():
        ps = list_images(root, split, cls, limit)
        paths.extend(ps)
        meta.extend([(cls, shape, color, p) for p in ps])

    encoded, integer_ok = encode_paths(paths, encoder)

    rows = []
    for (cls, shape, color, p), psi in zip(meta, encoded):
        rows.append({
            "class": cls,
            "shape": shape,
            "color": color,
            "path": p,
            "psi": psi,
        })

    return rows, integer_ok


def channel_energy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return 16 channel energies."""
    d = ring_distance_u8(a, b)
    return d.sum(axis=(1,2), dtype=np.int64)


def selected_energy(a: np.ndarray, b: np.ndarray, channels) -> int:
    d = ring_distance_u8(a[list(channels)], b[list(channels)])
    return int(d.sum(dtype=np.int64))


def class_rows(rows, cls):
    return [r for r in rows if r["class"] == cls]


def cross_class_channel_measure(A, B) -> np.ndarray:
    """Symmetric nearest-cross-class channel measure.

    Each state measures to its closest state in the other class independently
    per channel. Median over both directions avoids one pose dominating.
    """
    vals = []

    def one_way(src, dst):
        for a in src:
            best = np.full(16, np.iinfo(np.int64).max, dtype=np.int64)
            for b in dst:
                e = channel_energy(a["psi"], b["psi"])
                best = np.minimum(best, e)
            vals.append(best)

    one_way(A, B)
    one_way(B, A)
    return np.median(np.stack(vals, axis=0), axis=0)


def visible_reference_pairs(visible_classes):
    same_shape = []
    same_color = []

    for a, b in itertools.combinations(visible_classes, 2):
        sa, ca = CLASSES[a]
        sb, cb = CLASSES[b]

        if sa == sb and ca != cb:
            same_shape.append((a,b))
        if ca == cb and sa != sb:
            same_color.append((a,b))

    assert len(same_shape) == 1, same_shape
    assert len(same_color) == 1, same_color
    return same_shape[0], same_color[0]


def discover_channels(train_visible):
    visible_classes = sorted({r["class"] for r in train_visible})
    shape_pair, color_pair = visible_reference_pairs(visible_classes)

    Dshape = cross_class_channel_measure(
        class_rows(train_visible, shape_pair[0]),
        class_rows(train_visible, shape_pair[1]),
    )
    Dcolor = cross_class_channel_measure(
        class_rows(train_visible, color_pair[0]),
        class_rows(train_visible, color_pair[1]),
    )

    score = (Dcolor - Dshape) / (Dcolor + Dshape + 1.0)

    shape_rank = list(np.argsort(-score))
    color_rank = list(np.argsort(score))

    shape_channels = shape_rank[:TOP_K]
    color_channels = color_rank[:TOP_K]

    return {
        "shape_reference_pair": shape_pair,
        "color_reference_pair": color_pair,
        "Dshape": Dshape,
        "Dcolor": Dcolor,
        "score": score,
        "shape_rank": shape_rank,
        "color_rank": color_rank,
        "shape_channels": shape_channels,
        "color_channels": color_channels,
    }


def nearest_record(query, train_visible, channels):
    best = None

    for i, r in enumerate(train_visible):
        E = selected_energy(query["psi"], r["psi"], channels)
        key = (E, i)
        if best is None or key < best[0]:
            best = (key, r, E)

    return best[1], best[2]


def evaluate_fold(train, test, heldout_class):
    truth = CLASSES[heldout_class]

    train_visible = [r for r in train if r["class"] != heldout_class]
    test_heldout = [r for r in test if r["class"] == heldout_class]

    assert all((r["shape"],r["color"]) != truth for r in train_visible)

    disc = discover_channels(train_visible)
    sch = disc["shape_channels"]
    cch = disc["color_channels"]
    allch = list(range(16))

    shape_ok = 0
    color_ok = 0
    pair_ok = 0
    whole_shape_ok = 0
    whole_color_ok = 0

    cases = []

    for q in test_heldout:
        rs, Es = nearest_record(q, train_visible, sch)
        rc, Ec = nearest_record(q, train_visible, cch)
        rw, Ew = nearest_record(q, train_visible, allch)

        pred = (rs["shape"], rc["color"])
        whole = (rw["shape"], rw["color"])

        s_ok = pred[0] == truth[0]
        c_ok = pred[1] == truth[1]
        p_ok = pred == truth

        shape_ok += int(s_ok)
        color_ok += int(c_ok)
        pair_ok += int(p_ok)
        whole_shape_ok += int(whole[0] == truth[0])
        whole_color_ok += int(whole[1] == truth[1])

        cases.append({
            "query": q,
            "shape_match": rs,
            "color_match": rc,
            "whole_match": rw,
            "shape_energy": Es,
            "color_energy": Ec,
            "whole_energy": Ew,
            "pred": pred,
            "whole": whole,
            "truth": truth,
            "success": p_ok,
            "shape_success": s_ok,
            "color_success": c_ok,
        })

    channel_rows = []
    for k, name in enumerate(CHANNELS):
        channel_rows.append({
            "index": k,
            "name": name,
            "same_shape_diff_color_measure": float(disc["Dshape"][k]),
            "same_color_diff_shape_measure": float(disc["Dcolor"][k]),
            "factor_score": float(disc["score"][k]),
        })

    return {
        "heldout_class": heldout_class,
        "truth": {"shape":truth[0], "color":truth[1]},
        "test_images": len(test_heldout),
        "shape_reference_pair": list(disc["shape_reference_pair"]),
        "color_reference_pair": list(disc["color_reference_pair"]),
        "shape_channels": [{"index":k,"name":CHANNELS[k],"score":float(disc["score"][k])} for k in sch],
        "color_channels": [{"index":k,"name":CHANNELS[k],"score":float(disc["score"][k])} for k in cch],
        "channel_diagnostics": channel_rows,
        "shape_correct": shape_ok,
        "color_correct": color_ok,
        "constructive_pair_correct": pair_ok,
        "whole_memory_shape_correct": whole_shape_ok,
        "whole_memory_color_correct": whole_color_ok,
        "_cases": cases,
    }


def thumb(path, size=(145,145)):
    im = Image.open(path).convert("RGB")
    im.thumbnail(size)
    c = Image.new("RGB", size, "white")
    c.paste(im, ((size[0]-im.width)//2,(size[1]-im.height)//2))
    return c


def pick_visual_cases(fold):
    cases = fold["_cases"]
    success = [c for c in cases if c["success"]]
    shape_fail = [c for c in cases if not c["shape_success"] and c["color_success"]]
    color_fail = [c for c in cases if c["shape_success"] and not c["color_success"]]
    both_fail = [c for c in cases if not c["shape_success"] and not c["color_success"]]

    pick = []
    for bucket in (success, shape_fail, color_fail, both_fail):
        if bucket:
            pick.append(bucket[0])

    for c in cases:
        if len(pick) >= 4:
            break
        if all(c is not x for x in pick):
            pick.append(c)

    return pick[:4]


def make_photo_audit(folds, path):
    rows = []
    for f in folds:
        for c in pick_visual_cases(f):
            rows.append((f,c))

    W = 1240
    row_h = 205
    H = 80 + row_h * len(rows)

    sheet = Image.new("RGB", (W,H), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    d.text((20,15), "v23 — REAL PHOTO AUDIT USING OWNER'S 16-CHANNEL WAVE ENCODER", fill="black", font=font)
    d.text((20,35), "QUERY | nearest SHAPE-channel memory | nearest COLOR-channel memory | nearest ALL-16 memory", fill="black", font=font)
    d.text((20,53), "Green = constructed held-out factor pair exactly; Red = failure", fill="black", font=font)

    for i,(fold,c) in enumerate(rows):
        y = 80 + i*row_h
        border = (35,140,55) if c["success"] else (190,45,45)
        d.rectangle((8,y,1232,y+row_h-8), outline=border, width=3)

        panels = [
            ("QUERY", c["query"]),
            ("SHAPE NN", c["shape_match"]),
            ("COLOR NN", c["color_match"]),
            ("ALL-16 NN", c["whole_match"]),
        ]
        xs = [20,185,350,515]

        for x,(label,r) in zip(xs,panels):
            d.text((x,y+7), label, fill="black", font=font)
            sheet.paste(thumb(r["path"]), (x,y+27))
            d.text((x,y+176), r["class"], fill="black", font=font)

        tx = 685
        d.text((tx,y+8), f"held out: {fold['heldout_class']}", fill="black", font=font)
        d.text((tx,y+27), f"truth: {c['truth'][0]} + {c['truth'][1]}", fill="black", font=font)
        d.text((tx,y+46), f"constructed: {c['pred'][0]} + {c['pred'][1]}", fill=border, font=font)
        d.text((tx,y+70), f"shape E={c['shape_energy']}  color E={c['color_energy']}  all16 E={c['whole_energy']}", fill="black", font=font)

        sch = ",".join(str(x["name"]) for x in fold["shape_channels"])
        cch = ",".join(str(x["name"]) for x in fold["color_channels"])
        d.text((tx,y+94), f"shape channels: {sch}", fill="black", font=font)
        d.text((tx,y+113), f"color channels: {cch}", fill="black", font=font)

        status = (
            ("OK" if c["shape_success"] else "WRONG") +
            " shape | " +
            ("OK" if c["color_success"] else "WRONG") +
            " color"
        )
        d.text((tx,y+140), status, fill=border, font=font)

    sheet.save(path)


def make_channel_role_sheet(folds, path):
    W = 1200
    panel_w = 290
    H = 690
    sheet = Image.new("RGB",(W,H),"white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    d.text((20,15),"v23 — CHANNEL FACTOR DISCOVERY (training-only within each held-out fold)",fill="black",font=font)
    d.text((20,33),"score > 0 = shape-preserving across color; score < 0 = color-preserving across shape",fill="black",font=font)

    for fi,f in enumerate(folds):
        x0=10+fi*panel_w
        d.rectangle((x0,55,x0+panel_w-8,H-10),outline=(70,70,70),width=2)
        d.text((x0+8,65),f"HOLD OUT: {f['heldout_class']}",fill="black",font=font)
        d.text((x0+8,82),f"shape ref: {' vs '.join(f['shape_reference_pair'])}",fill="black",font=font)
        d.text((x0+8,99),f"color ref: {' vs '.join(f['color_reference_pair'])}",fill="black",font=font)

        rows=sorted(f["channel_diagnostics"], key=lambda r:-r["factor_score"])
        y=125
        for r in rows:
            s=r["factor_score"]
            label=f"{r['index']:02d} {r['name']:<10} {s:+.3f}"
            col=(25,105,170) if s>0 else ((175,70,40) if s<0 else (70,70,70))
            d.text((x0+8,y),label,fill=col,font=font)

            center=x0+210
            span=60
            d.line((center,y+5,center+int(max(-1,min(1,s))*span),y+5),fill=col,width=3)
            y+=29

    sheet.save(path)


def json_fold(f):
    return {k:v for k,v in f.items() if k != "_cases"}


def main(root, out_json, out_photos, out_channels):
    root = Path(root)
    encoder = QCMWaveEncoder().eval()

    train, int_train = build_records(root,"Training",TRAIN_PER_CLASS,encoder)
    test, int_test = build_records(root,"Test",TEST_PER_CLASS,encoder)

    folds = [evaluate_fold(train,test,cls) for cls in CLASSES]

    total = sum(f["test_images"] for f in folds)
    shape = sum(f["shape_correct"] for f in folds)
    color = sum(f["color_correct"] for f in folds)
    pair = sum(f["constructive_pair_correct"] for f in folds)
    whole_shape = sum(f["whole_memory_shape_correct"] for f in folds)
    whole_color = sum(f["whole_memory_color_correct"] for f in folds)

    score_matrix=np.asarray([
        [r["factor_score"] for r in f["channel_diagnostics"]]
        for f in folds
    ],dtype=float)

    stability=[]
    for k,name in enumerate(CHANNELS):
        vals=score_matrix[:,k]
        stability.append({
            "index":k,
            "name":name,
            "mean_score":float(vals.mean()),
            "min_score":float(vals.min()),
            "max_score":float(vals.max()),
            "shape_lean_folds":int((vals>0).sum()),
            "color_lean_folds":int((vals<0).sum()),
        })

    report = {
        "model":"v23-owner-16ch-wave-factor-discovery",
        "encoder":{
            "source":"owner-supplied scripts/wave_encoder_torch.py",
            "shape":[16,FH,FW],
            "channel_names":CHANNELS,
            "integer_output_train":bool(int_train),
            "integer_output_test":bool(int_test),
            "manual_channel_semantics_for_factor_selection":False,
        },
        "dataset":{
            "classes":CLASSES,
            "train_per_class":TRAIN_PER_CLASS,
            "test_per_class":TEST_PER_CLASS,
            "heldout_whole_combination_each_fold":True,
        },
        "factor_score_equation":"(D_color_invariant - D_shape_invariant)/(D_color_invariant + D_shape_invariant + 1)",
        "top_k_per_factor":TOP_K,
        "folds":[json_fold(f) for f in folds],
        "channel_stability":stability,
        "aggregate":{
            "test_images":total,
            "shape_correct":shape,
            "color_correct":color,
            "constructive_pair_correct":pair,
            "whole_memory_shape_correct":whole_shape,
            "whole_memory_color_correct":whole_color,
        },
        "claim_boundary":(
            "v23 is a real-image factor-discovery gate using the owner's fixed 16-channel "
            "wave encoder and Z256 ring MEASURE. The factor labels are used only to define "
            "the controlled 2x2 audit and training reference relations. It does not yet prove "
            "unsupervised semantic discovery or final Arshad-ViT attention."
        ),
    }

    out=Path(out_json)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")

    make_photo_audit(folds,out_photos)
    make_channel_role_sheet(folds,out_channels)

    print(json.dumps({
        "aggregate":report["aggregate"],
        "folds":[{
            "heldout":f["heldout_class"],
            "shape":[f["shape_correct"],f["test_images"]],
            "color":[f["color_correct"],f["test_images"]],
            "constructed":[f["constructive_pair_correct"],f["test_images"]],
            "shape_channels":[x["name"] for x in f["shape_channels"]],
            "color_channels":[x["name"] for x in f["color_channels"]],
        } for f in folds],
        "channel_stability":sorted(stability,key=lambda x:-abs(x["mean_score"])),
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--json",default="results/v23_wave_factor_discovery.json")
    ap.add_argument("--photos",default="results/v23_real_photo_audit.png")
    ap.add_argument("--channels",default="results/v23_channel_roles.png")
    a=ap.parse_args()
    main(a.root,a.json,a.photos,a.channels)
