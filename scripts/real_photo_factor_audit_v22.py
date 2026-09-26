"""v22 — Raw-photo audit for v21 orthogonal factor observer.

Creates actual photo contact sheets:

    QUERY
      -> nearest SHAPE-memory photo
      -> nearest COLOR-memory photo
      -> constructed factor pair

Examples are selected from each held-out fold with both successes and failures
where available.

This is diagnostic only. It does not change the observer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    if den<=0: return 0
    return int((int(num)*127 + den//2)//den)


def load_rgb(p):
    return np.asarray(Image.open(p).convert("RGB"),dtype=np.uint8)


def foreground(rgb):
    return np.any(rgb<245,axis=2)


def bbox(mask):
    ys,xs=np.where(mask)
    if len(xs)==0:
        return 0,0,mask.shape[1],mask.shape[0]
    return int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1


def band_occupancy127(z,axis,bands):
    groups=np.array_split(np.arange(z.shape[axis]),bands)
    out=[]
    for ids in groups:
        part=z[ids,:] if axis==0 else z[:,ids]
        out.append(ratio127(int(part.sum()),int(part.size)))
    return out


def row_boundaries127(z,bands=16):
    groups=np.array_split(np.arange(z.shape[0]),bands)
    W=z.shape[1]; den=max(1,W-1); out=[]
    for ids in groups:
        cols=np.where(np.any(z[ids,:],axis=0))[0]
        if len(cols)==0: out += [0,0]
        else: out += [ratio127(int(cols.min()),den),ratio127(int(cols.max()),den)]
    return out


def col_boundaries127(z,bands=7):
    groups=np.array_split(np.arange(z.shape[1]),bands)
    H=z.shape[0]; den=max(1,H-1); out=[]
    for ids in groups:
        rows=np.where(np.any(z[:,ids],axis=1))[0]
        if len(rows)==0: out += [0,0]
        else: out += [ratio127(int(rows.min()),den),ratio127(int(rows.max()),den)]
    return out


def shape_state(rgb):
    m=foreground(rgb)
    x0,y0,x1,y1=bbox(m)
    z=m[y0:y1,x0:x1]
    h,w=z.shape
    mx=max(h,w,1)
    out=[ratio127(w,mx),ratio127(h,mx),ratio127(int(z.sum()),int(z.size))]
    out+=band_occupancy127(z,0,16)
    out+=band_occupancy127(z,1,7)
    out+=row_boundaries127(z,16)
    out+=col_boundaries127(z,7)
    q=np.asarray(out,dtype=np.uint8)
    assert q.shape==(72,) and int(q.max())<=127
    return q


def trunc_div2_signed(x):
    x=np.asarray(x,dtype=np.int16)
    out=np.empty_like(x,dtype=np.int16)
    pos=x>=0
    out[pos]=x[pos]//2
    out[~pos]=-((-x[~pos])//2)
    return out


def signed_quantile(x,num,den):
    x=np.asarray(x,dtype=np.int16).reshape(-1)
    if len(x)==0: return 0
    k=((len(x)-1)*num)//den
    return int(np.partition(x,k)[k])


def chroma_encode(v):
    v=max(-127,min(127,int(v)))
    return (128-v)&255


def color_state(rgb):
    m=foreground(rgb)
    pix=rgb[m].astype(np.int16)
    if len(pix)==0:
        return np.full(18,128,dtype=np.uint8)
    R,G,B=pix[:,0],pix[:,1],pix[:,2]
    rels=(trunc_div2_signed(R-G),trunc_div2_signed(G-B),trunc_div2_signed(B-R))
    out=[]
    for d in rels:
        vals=[
            signed_quantile(d,1,4),
            signed_quantile(d,1,2),
            signed_quantile(d,3,4),
            int(d.astype(np.int64).sum()//len(d)),
        ]
        out += [chroma_encode(v) for v in vals]
    for d in rels:
        out += [ratio127(int((d>0).sum()),len(d)),ratio127(int((d<0).sum()),len(d))]
    q=np.asarray(out,dtype=np.uint8)
    assert q.shape==(18,)
    return q


def observe(p):
    rgb=load_rgb(p)
    return shape_state(rgb),color_state(rgb)


def list_images(root,split,cls,limit):
    d=Path(root)/split/cls
    return sorted([p for p in d.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"}])[:limit]


def build(root,split,limit):
    rows=[]
    for cls,(slabel,clabel) in CLASSES.items():
        for p in list_images(root,split,cls,limit):
            s,c=observe(p)
            rows.append({
                "class":cls,"shape_label":slabel,"color_label":clabel,
                "shape":s,"color":c,"path":p,
            })
    return rows


def memories(train,heldout_pair):
    sm=[];cm=[]
    for r in train:
        pair=(r["shape_label"],r["color_label"])
        if pair==heldout_pair:
            continue
        sm.append(r)
        cm.append(r)
    return sm,cm


def nearest_record(q,key,memory):
    best=None
    for i,r in enumerate(memory):
        E=energy(q,r[key])
        z=(E,i)
        if best is None or z<best[0]:
            best=(z,r,E)
    return best[1],best[2]


def collect_cases(train,test):
    cases=[]
    for heldout_class,truth in CLASSES.items():
        sm,cm=memories(train,truth)
        rows=[r for r in test if r["class"]==heldout_class]
        fold=[]

        for r in rows:
            rs,Es=nearest_record(r["shape"],"shape",sm)
            rc,Ec=nearest_record(r["color"],"color",cm)
            pred=(rs["shape_label"],rc["color_label"])
            ok=pred==truth
            fold.append({
                "query":r,
                "shape_match":rs,
                "color_match":rc,
                "shape_energy":Es,
                "color_energy":Ec,
                "pred":pred,
                "truth":truth,
                "success":ok,
            })

        successes=[x for x in fold if x["success"]]
        failures=[x for x in fold if not x["success"]]

        # Up to 2 success + 2 failure per fold.
        pick=successes[:2]+failures[:2]
        if len(pick)<4:
            rest=[x for x in fold if x not in pick]
            pick+=rest[:4-len(pick)]

        cases += pick

    return cases


def open_thumb(p,size=(150,150)):
    im=Image.open(p).convert("RGB")
    im.thumbnail(size)
    canvas=Image.new("RGB",size,"white")
    x=(size[0]-im.width)//2; y=(size[1]-im.height)//2
    canvas.paste(im,(x,y))
    return canvas


def draw_case_sheet(cases,out_path):
    W=1120
    row_h=230
    H=70+row_h*len(cases)
    sheet=Image.new("RGB",(W,H),"white")
    d=ImageDraw.Draw(sheet)
    font=ImageFont.load_default()

    d.text((20,20),"v22 Real-photo audit: QUERY -> SHAPE memory -> COLOR memory -> CONSTRUCT",fill="black",font=font)
    d.text((20,40),"Green border = exact constructed pair; red border = factor failure",fill="black",font=font)

    for i,c in enumerate(cases):
        y=70+i*row_h
        color=(40,140,60) if c["success"] else (180,40,40)
        d.rectangle((10,y,1110,y+row_h-10),outline=color,width=3)

        x0s=[25,205,385]
        labels=["QUERY","NEAREST SHAPE","NEAREST COLOR"]
        imgs=[
            c["query"]["path"],
            c["shape_match"]["path"],
            c["color_match"]["path"],
        ]

        for x0,label,p in zip(x0s,labels,imgs):
            d.text((x0,y+8),label,fill="black",font=font)
            sheet.paste(open_thumb(p),(x0,y+30))

        truth_shape,truth_color=c["truth"]
        pred_shape,pred_color=c["pred"]

        tx=570
        d.text((tx,y+10),f"Held out: {c['query']['class']}",fill="black",font=font)
        d.text((tx,y+30),f"Truth:       {truth_shape} + {truth_color}",fill="black",font=font)
        d.text((tx,y+50),f"Constructed: {pred_shape} + {pred_color}",fill=color,font=font)

        d.text((tx,y+80),f"Shape memory class: {c['shape_match']['class']}",fill="black",font=font)
        d.text((tx,y+100),f"Shape MEASURE: {c['shape_energy']}",fill="black",font=font)

        d.text((tx,y+130),f"Color memory class: {c['color_match']['class']}",fill="black",font=font)
        d.text((tx,y+150),f"Color MEASURE: {c['color_energy']}",fill="black",font=font)

        d.text((tx,y+180),f"Query file: {c['query']['path'].name}",fill="black",font=font)

    sheet.save(out_path)


def main(root,out_png,out_json):
    train=build(root,"Training",MAX_TRAIN_PER_CLASS)
    test=build(root,"Test",MAX_TEST_PER_CLASS)
    cases=collect_cases(train,test)

    draw_case_sheet(cases,out_png)

    rows=[]
    for c in cases:
        rows.append({
            "heldout_class":c["query"]["class"],
            "query_file":c["query"]["path"].name,
            "truth":{"shape":c["truth"][0],"color":c["truth"][1]},
            "constructed":{"shape":c["pred"][0],"color":c["pred"][1]},
            "success":c["success"],
            "nearest_shape":{
                "class":c["shape_match"]["class"],
                "file":c["shape_match"]["path"].name,
                "measure":c["shape_energy"],
            },
            "nearest_color":{
                "class":c["color_match"]["class"],
                "file":c["color_match"]["path"].name,
                "measure":c["color_energy"],
            },
        })

    p=Path(out_json)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({"cases":rows},indent=2),encoding="utf-8")
    print(json.dumps({"cases":rows},indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--png",default="results/v22_real_photo_audit.png")
    ap.add_argument("--json",default="results/v22_real_photo_audit.json")
    a=ap.parse_args()
    main(a.root,a.png,a.json)
