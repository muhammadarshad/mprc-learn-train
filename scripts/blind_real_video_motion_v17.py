"""v17 — Blind real-video MPRC motion benchmark.

Purpose
=======
Stop the synthetic hide-and-seek loop.

Input is an actual recorded video. MPRC is NOT told any transform.

MPRC path (integer only)
------------------------
1. Decode consecutive video frames.
2. Use one raw byte plane (green channel) directly; no float grayscale in MPRC path.
3. Find the most temporally active native byte rectangle among:
       16x7 and 7x16
   by ring cdist energy.
4. Around that rectangle, search integer displacement:
       dx,dy in [-7,+7]
   using exact Z256 circular-distance MEASURE.
5. Return best displacement, support, energy, second-best gap.

Independent benchmark only
--------------------------
OpenCV Farneback optical flow is computed separately and NEVER enters MPRC
selection. Its median flow over the selected MPRC rectangle is used only as an
external reference.

Reported:
- x/y direction agreement
- rounded-vector agreement
- endpoint error to external flow
- per-frame MPRC energy/confidence
- stationary vs moving cases
- native 16x7 vs 7x16 orientation chosen

This is not optical-flow ground truth. Farneback is an independent conventional
reference. A later benchmark with instrumented/annotated physical motion is still
required for ground-truth claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

SEARCH=7
NATIVE_SHAPES=((16,7),(7,16))
MAX_PAIRS=100
START_FRAME=20


def ring_cdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa=a.astype(np.int16,copy=False)
    bb=b.astype(np.int16,copy=False)
    ab=(aa-bb)&255
    ba=(bb-aa)&255
    return np.minimum(ab,ba)


def ring_energy(a,b)->int:
    return int(ring_cdist(a,b).sum(dtype=np.int64))


def best_motion_rectangle(prev_u8,curr_u8):
    """Choose most changing non-overlapping native 16x7 or 7x16 rectangle."""
    H,W=prev_u8.shape
    best=None

    for h,w in NATIVE_SHAPES:
        for y in range(0,H-h+1,h):
            for x in range(0,W-w+1,w):
                A=prev_u8[y:y+h,x:x+w]
                B=curr_u8[y:y+h,x:x+w]
                E=ring_energy(A,B)

                # deterministic: max energy, then top-left, then 16x7 before 7x16
                key=(-E,y,x,0 if (h,w)==(16,7) else 1)
                if best is None or key<best[0]:
                    best=(key,y,x,h,w,E)

    _,y,x,h,w,E=best
    return y,x,h,w,E


def estimate_translation(prev_u8,curr_u8,y,x,h,w):
    """Search actual next-frame location of the selected previous-frame rectangle."""
    H,W=prev_u8.shape
    ref=prev_u8[y:y+h,x:x+w]

    rows=[]
    for dy in range(-SEARCH,SEARCH+1):
        yy=y+dy
        if yy<0 or yy+h>H:
            continue

        for dx in range(-SEARCH,SEARCH+1):
            xx=x+dx
            if xx<0 or xx+w>W:
                continue

            cand=curr_u8[yy:yy+h,xx:xx+w]
            E=ring_energy(ref,cand)
            exact=int((ref==cand).sum())

            # MEASURE primary. Exact support secondary. Small movement is only tie-break.
            key=(E,-exact,abs(dx)+abs(dy),abs(dx),abs(dy),dy,dx)
            rows.append((key,dx,dy,E,exact))

    rows.sort(key=lambda z:z[0])
    best=rows[0]
    second=rows[1] if len(rows)>1 else rows[0]

    _,dx,dy,E,exact=best
    gap=int(second[3]-E)
    return {
        "dx":int(dx),
        "dy":int(dy),
        "energy":int(E),
        "exact_support":int(exact),
        "site_count":int(h*w),
        "second_energy_gap":gap,
    }


def farneback_reference(prev_bgr,curr_bgr,y,x,h,w):
    """External benchmark only; not part of the MPRC computation."""
    g0=cv2.cvtColor(prev_bgr,cv2.COLOR_BGR2GRAY)
    g1=cv2.cvtColor(curr_bgr,cv2.COLOR_BGR2GRAY)

    flow=cv2.calcOpticalFlowFarneback(
        g0,g1,None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    roi=flow[y:y+h,x:x+w]
    fx=float(np.median(roi[:,:,0]))
    fy=float(np.median(roi[:,:,1]))
    return fx,fy


def sign_eps(x,eps=0.25):
    if x>eps: return 1
    if x<-eps: return -1
    return 0


def main(video_path:str,out_path:str):
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES,START_FRAME)
    ok,prev=cap.read()
    if not ok:
        raise RuntimeError("cannot read start frame")

    rows=[]
    pair=0

    while pair<MAX_PAIRS:
        ok,curr=cap.read()
        if not ok:
            break

        # MPRC path: raw byte plane only.
        p=prev[:,:,1]
        c=curr[:,:,1]

        y,x,h,w,temporal_E=best_motion_rectangle(p,c)
        m=estimate_translation(p,c,y,x,h,w)

        # Independent conventional reference after MPRC decision is fixed.
        fx,fy=farneback_reference(prev,curr,y,x,h,w)

        mdx=int(m["dx"]); mdy=int(m["dy"])
        rdx=int(np.rint(fx)); rdy=int(np.rint(fy))

        sx_match=sign_eps(mdx)==sign_eps(fx)
        sy_match=sign_eps(mdy)==sign_eps(fy)
        vec_round_match=(mdx==rdx and mdy==rdy)
        endpoint_error=math.sqrt((mdx-fx)**2+(mdy-fy)**2)

        rows.append({
            "pair_index":pair,
            "frame0":START_FRAME+pair,
            "frame1":START_FRAME+pair+1,
            "native_shape":[h,w],
            "rect":[x,y,w,h],
            "temporal_ring_energy":int(temporal_E),
            "mprc":{
                **m,
            },
            "external_farneback":{
                "dx":fx,
                "dy":fy,
                "rounded_dx":rdx,
                "rounded_dy":rdy,
            },
            "comparison":{
                "x_sign_match":bool(sx_match),
                "y_sign_match":bool(sy_match),
                "both_sign_match":bool(sx_match and sy_match),
                "rounded_vector_match":bool(vec_round_match),
                "endpoint_error":float(endpoint_error),
            },
        })

        prev=curr
        pair+=1

    cap.release()
    if len(rows)<10:
        raise AssertionError("too few real frame pairs")

    n=len(rows)
    both=sum(int(r["comparison"]["both_sign_match"]) for r in rows)
    xv=sum(int(r["comparison"]["x_sign_match"]) for r in rows)
    yv=sum(int(r["comparison"]["y_sign_match"]) for r in rows)
    rv=sum(int(r["comparison"]["rounded_vector_match"]) for r in rows)
    epe=[r["comparison"]["endpoint_error"] for r in rows]

    shape_counts={}
    for r in rows:
        k=f'{r["native_shape"][0]}x{r["native_shape"][1]}'
        shape_counts[k]=shape_counts.get(k,0)+1

    # Confidence diagnostic: exact ring-measure gap above zero means unique-ish best candidate.
    positive_gap=sum(int(r["mprc"]["second_energy_gap"]>0) for r in rows)

    report={
        "model":"v17-blind-real-video-mprc-motion",
        "source":{
            "video":video_path,
            "frame_count":frame_count,
            "fps":fps,
            "width":width,
            "height":height,
            "start_frame":START_FRAME,
            "pairs_used":n,
            "movement_injected_by_experiment":False,
        },
        "mprc_path":{
            "storage":"raw uint8 green plane",
            "native_rectangles":["16x7","7x16"],
            "distance":"Z256 circular distance",
            "search_dx":[-SEARCH,SEARCH],
            "search_dy":[-SEARCH,SEARCH],
            "float_in_mprc_path":False,
            "labels_used":False,
        },
        "external_reference":{
            "method":"OpenCV Farneback dense optical flow",
            "used_inside_mprc_path":False,
            "role":"independent comparison only; not ground truth",
        },
        "benchmark":{
            "x_direction_agreement":[xv,n],
            "y_direction_agreement":[yv,n],
            "both_direction_agreement":[both,n],
            "rounded_vector_agreement":[rv,n],
            "mean_endpoint_error":float(np.mean(epe)),
            "median_endpoint_error":float(np.median(epe)),
            "positive_second_best_gap":[positive_gap,n],
            "native_shape_counts":shape_counts,
        },
        "frames":rows,
        "claim_boundary":(
            "This uses naturally occurring motion in recorded video and does not inject a "
            "known transform. Farneback is an independent conventional reference, not physical "
            "ground truth. Agreement supports blind real-data motion inference; disagreement "
            "does not by itself identify which estimator is correct."
        ),
    }

    out=Path(out_path)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print(json.dumps({
        "pairs":n,
        "both_direction_agreement":[both,n],
        "rounded_vector_agreement":[rv,n],
        "mean_endpoint_error":float(np.mean(epe)),
        "median_endpoint_error":float(np.median(epe)),
        "positive_second_best_gap":[positive_gap,n],
        "native_shape_counts":shape_counts,
    },indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--video",required=True)
    ap.add_argument("--out",default="results/blind_real_video_motion_v17.json")
    args=ap.parse_args()
    main(args.video,args.out)
