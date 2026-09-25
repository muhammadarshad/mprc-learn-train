"""v4 — Local directional credit events, no backpropagated activation gradient.

v3 established that sign/event-only D4 learning reaches substantial accuracy, but
its LUT event sign still came from:

    raw_credit = dscores @ W.T

Even with magnitude discarded, that is a conventional backward matrix credit path.

v4 removes it.

For every violated target y vs rival c and every feature f:

READOUT EVENT
-------------
a_f = 128 - LUT[p_f]
sign(a_f) alone determines how W[f,y] and W[f,c] should move:

    W[f,y] += sign(a_f)
    W[f,c] -= sign(a_f)

LUT EVENT
---------
The local target-vs-rival weight difference determines whether increasing or
decreasing activation improves the class gap:

    gap_f = W[f,y] - W[f,c]

Since a_f = 128 - LUT[p_f]:

    LUT[p_f] += -sign(gap_f)

No dscores@W.T, no activation derivative, no gradient magnitude.

All requested movements are counted in four independent residual memories:
    (+U,+D,-U,-D)
and emit only after DEN=1000 events.

Controlled branches:
  E-BP    : v3 sign-event rule with backward matrix credit (reference)
  E-LOCAL : local target/rival event rule, no backward matrix credit
  E-WONLY : local target/rival readout events, LUT frozen

This remains a research probe, not a frozen MPRC theorem.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits

DEN=1000
SEED=20260925
EPOCHS=100
BATCH=64
MARGIN=16
NCLASS=10


def signed_coord(v,ring):
    x=np.asarray(v,dtype=np.int64)
    return (128-(x&255)) if ring else x


class D4Param:
    def __init__(self,value,ring=False):
        self.value=np.asarray(value,dtype=np.int64).copy()
        self.ring=bool(ring)
        self.residual=np.zeros(self.value.shape+(4,),dtype=np.int64)
        self.emitted=np.zeros(4,dtype=np.int64)

    def step_delta(self,req):
        req=np.asarray(req,dtype=np.int64)
        mag=np.abs(req)
        coord=signed_coord(self.value,self.ring)
        up=req>0
        down=req<0
        plus=coord>=0
        minus=~plus
        masks=(plus&up,plus&down,minus&up,minus&down)
        delta=np.zeros_like(self.value)
        for d,mask in enumerate(masks):
            if not np.any(mask):
                continue
            r=self.residual[...,d]
            r[mask]+=mag[mask]
            k=np.zeros_like(r)
            k[mask]=r[mask]//DEN
            r[mask]-=k[mask]*DEN
            self.emitted[d]+=int(k.sum())
            if d in (0,2):
                delta+=k
            else:
                delta-=k
        self.value+=delta
        if self.ring:
            self.value&=255
        assert np.all((self.residual>=0)&(self.residual<DEN))


class Net:
    def __init__(self,fdim,seed):
        rng=np.random.default_rng(seed)
        self.L=D4Param(np.arange(256,dtype=np.int64),ring=True)
        self.W=D4Param(rng.integers(-2,3,size=(fdim,NCLASS),dtype=np.int64))
        self.b=D4Param(np.zeros(NCLASS,dtype=np.int64))
        self.L0=self.L.value.copy()

    def forward(self,idx):
        ring=self.L.value[idx]
        a=128-ring
        return a,a@self.W.value+self.b.value

    def eval(self,idx,y):
        _,s=self.forward(idx)
        p=s.argmax(axis=1)
        return int((p==y).sum()),len(y)

    def violated_pairs(self,s,y):
        pairs=[]
        for i in range(len(y)):
            yi=int(y[i]); sy=int(s[i,yi])
            for c in range(NCLASS):
                if c==yi:
                    continue
                if int(s[i,c])+MARGIN>sy:
                    pairs.append((i,yi,c))
        return pairs

    def updates_bp_sign(self,idx,a,s,y):
        # v3 event-only reference. Magnitudes are removed, but LUT credit still
        # obtains its direction through matrix propagation.
        ds=np.zeros_like(s,dtype=np.int64)
        for i,yi,c in self.violated_pairs(s,y):
            ds[i,c]+=1
            ds[i,yi]-=1

        sa=np.sign(a).astype(np.int64)
        dW=-(sa.T@ds)
        db=-ds.sum(axis=0,dtype=np.int64)

        raw=ds@self.W.value.T
        ev=np.sign(raw).astype(np.int64)
        dL=np.zeros(256,dtype=np.int64)
        np.add.at(dL,idx.reshape(-1),ev.reshape(-1))
        return dW,db,dL,int(np.abs(ds).sum()//2)

    def updates_local(self,idx,a,s,y,lut_enabled=True):
        # Pure local target/rival events. No ds matrix and no W.T propagation.
        dW=np.zeros_like(self.W.value)
        db=np.zeros_like(self.b.value)
        dL=np.zeros(256,dtype=np.int64)
        pairs=self.violated_pairs(s,y)

        for i,yi,c in pairs:
            sign_a=np.sign(a[i]).astype(np.int64)

            # Increase target and decrease rival using direction only.
            dW[:,yi]+=sign_a
            dW[:,c]-=sign_a
            db[yi]+=1
            db[c]-=1

            if lut_enabled:
                # Local class-pair credit at each feature.
                gap=self.W.value[:,yi]-self.W.value[:,c]
                # a = 128-L, therefore improving gap requests:
                # delta L = -sign(weight gap)
                lut_req=-np.sign(gap).astype(np.int64)
                np.add.at(dL,idx[i],lut_req)

        return dW,db,dL,len(pairs)

    def train(self,Xtr,ytr,Xte,yte,mode):
        rng=np.random.default_rng(SEED+700+{"E-BP":0,"E-LOCAL":1,"E-WONLY":2}[mode])
        hist=[]
        best=(0,0)
        for epoch in range(EPOCHS):
            order=rng.permutation(len(Xtr))
            violations=0

            for st in range(0,len(order),BATCH):
                ids=order[st:st+BATCH]
                idx=Xtr[ids]; yy=ytr[ids]
                a,s=self.forward(idx)

                if mode=="E-BP":
                    dW,db,dL,v=self.updates_bp_sign(idx,a,s,yy)
                elif mode=="E-LOCAL":
                    dW,db,dL,v=self.updates_local(idx,a,s,yy,True)
                elif mode=="E-WONLY":
                    dW,db,dL,v=self.updates_local(idx,a,s,yy,False)
                else:
                    raise KeyError(mode)

                violations+=v
                self.W.step_delta(dW)
                self.b.step_delta(db)
                if mode!="E-WONLY":
                    self.L.step_delta(dL)

                self.W.value=np.clip(self.W.value,-127,127)
                self.b.value=np.clip(self.b.value,-32767,32767)

            tr=self.eval(Xtr,ytr)
            te=self.eval(Xte,yte)
            if te[0]>best[0]:
                best=(te[0],epoch+1)
            hist.append({
                "epoch":epoch+1,
                "violations":int(violations),
                "train_correct":tr[0],"train_total":tr[1],
                "test_correct":te[0],"test_total":te[1],
            })
            print(mode,hist[-1],flush=True)

        return {
            "final_test":list(self.eval(Xte,yte)),
            "best_test":[best[0],len(yte)],
            "best_epoch":best[1],
            "lut_changed":int(np.count_nonzero(self.L.value!=self.L0)),
            "emitted":{
                "LUT":self.L.emitted.tolist(),
                "W":self.W.emitted.tolist(),
                "b":self.b.emitted.tolist(),
            },
            "history":hist,
        }


def local_credit_sign_gate():
    # Exhaustively verify the LUT direction rule on simple one-feature class gaps.
    # Score gap G=a*(wy-wc), a=128-L. A one-step requested L movement should never
    # decrease G when the local rule is applicable away from the wrap boundary.
    tested=0
    for L in range(1,255):
        a=128-L
        for wy in range(-4,5):
            for wc in range(-4,5):
                gap=wy-wc
                if gap==0:
                    continue
                dL=-int(np.sign(gap))
                L2=L+dL
                a2=128-L2
                G=a*gap
                G2=a2*gap
                assert G2>G
                tested+=1
    return {"pass":True,"cases":tested}


gate=local_credit_sign_gate()

ds=load_digits()
X=(np.asarray(ds.data,dtype=np.int64)*15).astype(np.uint8)
y=np.asarray(ds.target,dtype=np.int64)

rng=np.random.default_rng(SEED)
order=rng.permutation(len(X))
cut=int(len(order)*.7)
trix=order[:cut];teix=order[cut:]
Xtr=X[trix];ytr=y[trix]
Xte=X[teix];yte=y[teix]

results={}
for mode in ("E-BP","E-LOCAL","E-WONLY"):
    # identical initial parameters for all branches
    net=Net(Xtr.shape[1],SEED+88)
    res=net.train(Xtr,ytr,Xte,yte,mode)
    results[mode]=res
    print(mode,"final",res["final_test"],"best",res["best_test"],"epoch",res["best_epoch"],flush=True)

report={
    "model":"v4-local-directional-credit-events",
    "gate":gate,
    "denominator":DEN,
    "bins":["+U","+D","-U","-D"],
    "softmax":False,
    "float_learning_rate":False,
    "activation_derivative":False,
    "branches":{
        "E-BP":"sign/event D4 with dscores@W.T only for LUT direction",
        "E-LOCAL":"no backward matrix credit; local target/rival weight-gap sign drives LUT events",
        "E-WONLY":"same local target/rival weight events with LUT frozen",
    },
    "results":results,
    "claim_boundary":(
        "E-LOCAL removes the backward matrix credit path and gradient magnitude. "
        "It remains supervised mistake-driven credit assignment because target/rival "
        "labels define the events. Success therefore supports a local discrete learning "
        "mechanism, not an unsupervised or theorem-level replacement of all backpropagation."
    )
}

root=Path(__file__).resolve().parents[1]
out=root/"results"/"local_directional_credit_events_v4.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2),flush=True)
