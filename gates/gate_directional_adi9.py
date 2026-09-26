#!/usr/bin/env python3
"""Exact survival gate for the corrected directional ADI-9 context.

Context order:
    C,U1,U2,D1,D2,F1,F2,B1,B2

No dataset, no training, no NumPy, no floating point.
"""
import math, json
from pathlib import Path

MOD=256
INV9=57

def enc(a):
    c=a[0]&255
    return [sum(a)&255]+[((c-a[k])&255) for k in range(1,9)]

def dec(z):
    c=(INV9*((z[0]+sum(z[1:]))&255))&255
    return [c]+[((c-d)&255) for d in z[1:]]

def s5_direct(a):
    return (a[0]+a[1]+a[3]+a[5]+a[7])&255

def s5_adi(z):
    c=dec(z)[0]
    return (5*c-z[1]-z[3]-z[5]-z[7])&255

def cont_direct(a):
    return ((a[2]-a[1])&255,(a[4]-a[3])&255,(a[6]-a[5])&255,(a[8]-a[7])&255)

def cont_adi(z):
    return ((z[1]-z[2])&255,(z[3]-z[4])&255,(z[5]-z[6])&255,(z[7]-z[8])&255)

assert (9*INV9)&255 == 1

basis=0
for coord in range(9):
    for v in range(256):
        a=[0]*9
        a[coord]=v
        z=enc(a)
        assert dec(z)==a
        assert s5_adi(z)==s5_direct(a)
        assert cont_adi(z)==cont_direct(a)
        basis+=1

seed=[0,17,33,65,129,7,91,200,255]
z0=enc(seed)
affine=0
for s in range(1,256,2):
    for t in range(256):
        a=[(s*x+t)&255 for x in seed]
        z=enc(a)
        expected=[(s*z0[0]+9*t)&255]+[((s*d)&255) for d in z0[1:]]
        assert z==expected
        assert dec(z)==a
        affine+=1

report={
    "gate":"directional_adi9",
    "status":"PASS",
    "ring":"Z_256",
    "context":["C","U1","U2","D1","D2","F1","F2","B1","B2"],
    "inv9":57,
    "basis_value_roundtrips":basis,
    "odd_affine_bind_cases":affine,
    "s5_identity":"5*C-(dU1+dD1+dF1+dB1)",
    "continuation":["dU1-dU2","dD1-dD2","dF1-dF2","dB1-dB2"],
}
out=Path("results/directional_adi9_survival.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
