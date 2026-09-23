#!/usr/bin/env python3
"""
CIFAR-10 exact active-byte codec using the frozen Paper-1
Quantinion Simplex Cubic Decomposition recovery mechanism.

This is a STORAGE/RECOVERY experiment, not a classifier.

Frozen theorem specialization:
    n = 8
    r = n+1 = 9
    R = Z_256
    72 atom coordinates
    ordered cubic observation has 8^3 = 512 coordinates

Codec:
    63 arbitrary payload bytes = 504 bits
      -> 72 atom bytes with canonical simplex parity fixed
      -> 72 rank-independent cubic coordinates
      -> remove their 72 known parity bits
      -> pack the remaining 504 bits into 63 active bytes
      -> restore parity
      -> recover atom bytes by seven exact one-bit lifts
      -> recover original 63 payload bytes

The theorem alone is information-preserving, not compressive:
63 payload bytes -> 63 packed active bytes.
Any compression must come from a higher-level MPRC shape/context model that
chooses fewer independent payload bytes to store.

Source theorem/verifier:
Five-Paper Research Booklet RC2, Paper 1, and
verify_quantinion_simplex_cubic.py.
"""

from pathlib import Path
import hashlib, json, pickle, tarfile, urllib.request, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/"data"/"cifar10"
CACHE.mkdir(parents=True,exist_ok=True)

URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE=CACHE/"cifar-10-python.tar.gz"
MD5="c58f30108f718f92721af3b95e74349a"

N=8
R=9
U=N*R            # 72 unknown atom-byte coordinates
PAYLOAD=63       # 504 free bits because each atom parity bit is fixed
CHUNK_BLOCKS=16384

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
    data=d[b"data"].astype(np.uint8)  # [N,3072], exact official byte order
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

# ------------------------------------------------------------------
# Exact Paper-1 GF(2) pieces
# ------------------------------------------------------------------
def gf2_rank(rows):
    basis={}
    rank=0
    for x0 in rows:
        x=x0
        while x:
            p=x.bit_length()-1
            if p in basis:
                x ^= basis[p]
            else:
                basis[p]=x
                rank+=1
                break
    return rank

def gf2_solve_rows(rows, rhs, ncols=72):
    # rows: list[int] bit-packed equation rows; rhs list 0/1.
    a=[[rows[i], rhs[i]&1] for i in range(len(rows))]
    pivot_for={}
    r=0
    # Gaussian elimination with integer bit rows.
    for c in range(ncols):
        p=next((i for i in range(r,len(a)) if (a[i][0]>>c)&1),None)
        if p is None:
            continue
        a[r],a[p]=a[p],a[r]
        for i in range(len(a)):
            if i!=r and ((a[i][0]>>c)&1):
                a[i][0]^=a[r][0]
                a[i][1]^=a[r][1]
        pivot_for[c]=r
        r+=1
    if r!=ncols:
        raise AssertionError(f"non-unique GF(2) system rank={r}/{ncols}")
    for bits,b in a[r:]:
        if bits==0 and b:
            raise AssertionError("inconsistent GF(2) system")
    x=[0]*ncols
    for c,row_idx in pivot_for.items():
        x[c]=a[row_idx][1]
    return x

def canonical_simplex():
    out=[[1]*N]
    for i in range(N):
        e=[0]*N;e[i]=1;out.append(e)
    return out

SUPPORTS=np.asarray(canonical_simplex(),dtype=np.uint8)
SUPPORT_FLAT=SUPPORTS.reshape(-1)

def tensor_sum_python(atoms,mod):
    t=[0]*(N*N*N)
    for v in atoms:
        z=0
        for i in range(N):
            for j in range(N):
                vij=v[i]*v[j]
                for k in range(N):
                    t[z]=(t[z]+vij*v[k])%mod
                    z+=1
    return t

def build_jacobian():
    supports=canonical_simplex()
    rows=[]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                bits=0
                col=0
                for q,s in enumerate(supports):
                    for a in range(N):
                        value=0
                        if i==a:value ^= s[j]&s[k]
                        if j==a:value ^= s[i]&s[k]
                        if k==a:value ^= s[i]&s[j]
                        if value:bits |= 1<<col
                        col+=1
                rows.append(bits)
    return rows

JINT=build_jacobian()
assert gf2_rank(JINT)==72

def independent_rows():
    basis={}
    selected=[]
    for idx,x0 in enumerate(JINT):
        x=x0
        while x:
            p=x.bit_length()-1
            if p in basis:
                x^=basis[p]
            else:
                basis[p]=x
                selected.append(idx)
                break
        if len(selected)==72:
            return selected
    raise AssertionError("failed to select rank-72 rows")

ACTIVE=independent_rows()
J72=[JINT[i] for i in ACTIVE]
assert gf2_rank(J72)==72

# Selected tensor coordinate triples.
II=np.asarray([idx//64 for idx in ACTIVE],dtype=np.int64)
JJ=np.asarray([(idx//8)%8 for idx in ACTIVE],dtype=np.int64)
KK=np.asarray([idx%8 for idx in ACTIVE],dtype=np.int64)

T2=tensor_sum_python(canonical_simplex(),2)
PARITY=np.asarray([T2[i]&1 for i in ACTIVE],dtype=np.uint8)

# Build columns of the inverse lifting Jacobian.
INV_COLS=[]
for r in range(72):
    rhs=[0]*72;rhs[r]=1
    x=gf2_solve_rows(J72,rhs)
    lo=0;hi=0
    for j,b in enumerate(x):
        if b:
            if j<64:lo|=1<<j
            else:hi|=1<<(j-64)
    INV_COLS.append((lo,hi))

# 9-byte packed rhs -> packed 72-bit delta lookup.
LUT_LO=np.zeros((9,256),dtype=np.uint64)
LUT_HI=np.zeros((9,256),dtype=np.uint64)
for pos in range(9):
    for value in range(256):
        lo=0;hi=0
        for bit in range(8):
            if value&(1<<bit):
                a,b=INV_COLS[pos*8+bit]
                lo^=a;hi^=b
        LUT_LO[pos,value]=lo
        LUT_HI[pos,value]=hi

BITPOS_LO=np.arange(64,dtype=np.uint64)
BITPOS_HI=np.arange(8,dtype=np.uint64)
WEIGHTS7=(1<<np.arange(7,dtype=np.uint16))[None,None,:]

# ------------------------------------------------------------------
# Batch codec
# ------------------------------------------------------------------
def payload_to_atoms(blocks):
    bits=np.unpackbits(blocks,axis=1,bitorder="little")
    seven=bits.reshape(len(blocks),72,7).astype(np.uint16)
    hi=(seven*WEIGHTS7).sum(axis=2,dtype=np.uint16)
    return (hi*2+SUPPORT_FLAT[None,:]).reshape(len(blocks),9,8).astype(np.uint16)

def selected_tensor(atoms,mod=256):
    a=atoms[:,:,II]
    b=atoms[:,:,JJ]
    c=atoms[:,:,KK]
    prod=(a*b*c).astype(np.uint32)
    return (prod.sum(axis=1,dtype=np.uint32)%mod).astype(np.uint16)

def selected_to_active(vals):
    hi=(vals>>1).astype(np.uint8)
    bits=((hi[:,:,None]>>np.arange(7,dtype=np.uint8)[None,None,:])&1)
    return np.packbits(bits.reshape(len(vals),504),axis=1,bitorder="little")

def active_to_selected(active):
    bits=np.unpackbits(active,axis=1,bitorder="little").reshape(len(active),72,7).astype(np.uint16)
    hi=(bits*WEIGHTS7).sum(axis=2,dtype=np.uint16)
    return (hi*2+PARITY[None,:]).astype(np.uint16)

def solve_delta(rhs):
    packed=np.packbits(rhs.astype(np.uint8),axis=1,bitorder="little")
    lo=np.zeros(len(rhs),dtype=np.uint64)
    hi=np.zeros(len(rhs),dtype=np.uint64)
    for p in range(9):
        lo ^= LUT_LO[p,packed[:,p]]
        hi ^= LUT_HI[p,packed[:,p]]
    out=np.empty((len(rhs),72),dtype=np.uint8)
    out[:,:64]=((lo[:,None]>>BITPOS_LO[None,:])&1).astype(np.uint8)
    out[:,64:]=((hi[:,None]>>BITPOS_HI[None,:])&1).astype(np.uint8)
    return out

def recover_atoms(selected):
    cur=np.broadcast_to(SUPPORT_FLAT,(len(selected),72)).astype(np.uint16).copy().reshape(len(selected),9,8)
    for b in range(1,8):
        mod=1<<(b+1);step=1<<b
        pred=selected_tensor(cur,mod)
        d=(selected.astype(np.int32)-pred.astype(np.int32))%mod
        if not np.all((d%step)==0):
            raise AssertionError("non-divisible lifting residual")
        rhs=((d//step)&1).astype(np.uint8)
        delta=solve_delta(rhs).reshape(len(selected),9,8).astype(np.uint16)
        cur=((cur+step*delta)%mod).astype(np.uint16)
    return cur.astype(np.uint8)

def atoms_to_payload(atoms):
    flat=atoms.reshape(len(atoms),72)
    if not np.all((flat&1)==SUPPORT_FLAT[None,:]):
        raise AssertionError("recovered simplex parity mismatch")
    hi=(flat>>1).astype(np.uint8)
    bits=((hi[:,:,None]>>np.arange(7,dtype=np.uint8)[None,None,:])&1)
    return np.packbits(bits.reshape(len(atoms),504),axis=1,bitorder="little")

def roundtrip_blocks(blocks):
    atoms=payload_to_atoms(blocks)
    vals=selected_tensor(atoms,256)
    active=selected_to_active(vals)
    if active.shape[1]!=63:
        raise AssertionError(active.shape)
    vals2=active_to_selected(active)
    got=atoms_to_payload(recover_atoms(vals2))
    return active,got

def run_stream(name,data):
    """
    Encode the exact dataset byte stream. Only the final global block is zero-padded.
    This tests storage without per-image block padding.
    """
    raw=data.reshape(-1)
    raw_bytes=int(raw.size)
    pad=(-raw_bytes)%63
    if pad:
        stream=np.concatenate([raw,np.zeros(pad,dtype=np.uint8)])
    else:
        stream=raw
    blocks=stream.reshape(-1,63)

    active_hasher=hashlib.sha256()
    recovered_hasher=hashlib.sha256()
    source_hasher=hashlib.sha256(raw.tobytes())
    mismatches=0
    t0=time.perf_counter()

    for start in range(0,len(blocks),CHUNK_BLOCKS):
        stop=min(len(blocks),start+CHUNK_BLOCKS)
        chunk=blocks[start:stop]
        active,got=roundtrip_blocks(chunk)
        active_hasher.update(active.tobytes())
        recovered_hasher.update(got.tobytes())
        mismatches += int(np.count_nonzero(got!=chunk))

    seconds=time.perf_counter()-t0
    # Decode stream hash includes padding; compare exact payload separately by rerun hash prefix is awkward,
    # so mismatch count is the primary byte-exact gate.
    return {
        "name":name,
        "images":int(len(data)),
        "raw_bytes":raw_bytes,
        "padding_bytes":int(pad),
        "blocks_63":int(len(blocks)),
        "packed_active_bytes":int(len(blocks)*63),
        "storage_ratio_active_over_raw":float((len(blocks)*63)/raw_bytes),
        "byte_mismatches_in_padded_stream":mismatches,
        "exact_roundtrip":bool(mismatches==0),
        "seconds":seconds,
        "payload_MB_per_s":float(raw_bytes/1e6/seconds),
        "source_sha256":source_hasher.hexdigest(),
        "active_sha256":active_hasher.hexdigest(),
        "recovered_padded_sha256":recovered_hasher.hexdigest(),
    }

def image_local_accounting(nimages):
    blocks_per_image=(3072+62)//63
    active=blocks_per_image*63
    return {
        "raw_bytes_per_image":3072,
        "blocks_per_image":blocks_per_image,
        "padding_bytes_per_image":active-3072,
        "active_bytes_per_image":active,
        "storage_ratio_active_over_raw":active/3072,
    }

# Exact structural redundancy facts.
from collections import Counter
row_mult=Counter(JINT)
basis_mult=Counter(row_mult[JINT[i]] for i in ACTIVE)

train_x,train_y,test_x,test_y=load_cifar()
assert train_x.shape==(50000,3072)
assert test_x.shape==(10000,3072)

train_result=run_stream("train",train_x)
print(json.dumps(train_result,indent=2),flush=True)
test_result=run_stream("test",test_x)
print(json.dumps(test_result,indent=2),flush=True)

result={
    "dataset":{
        "name":"CIFAR-10",
        "official_train_images":50000,
        "official_test_images":10000,
        "bytes_per_image":3072,
        "archive_md5":MD5,
    },
    "paper1_codec":{
        "n":8,
        "r":9,
        "atom_coordinates":72,
        "ordered_cubic_coordinates":512,
        "rank_selected_coordinates":72,
        "payload_bytes_per_block":63,
        "packed_active_bytes_per_block":63,
        "fixed_parity_bits_per_block":72,
        "lift_sequence":[2,4,8,16,32,64,128,256],
        "selected_coordinate_indices":ACTIVE,
        "basis_witness_multiplicity":{
            "1":int(basis_mult[1]),
            "3":int(basis_mult[3]),
            "6":int(basis_mult[6]),
        },
        "unique_irreplaceable_diagonal_equations":8,
    },
    "information_accounting":{
        "payload_bits":504,
        "free_atom_bits":504,
        "rank_selected_cubic_free_bits":504,
        "theorem_alone_is_compression":False,
        "theorem_only_storage_ratio":1.0,
    },
    "stream_results":{
        "train":train_result,
        "test":test_result,
    },
    "image_local_accounting":image_local_accounting(1),
    "interpretation":{
        "exact_recovery_role":"The cubic lift is an exact recoverable representation for bytes embedded in the fixed simplex structure.",
        "compression_role":"Actual image compression requires a higher-level MPRC shape/context rule that decides which source bytes are independent/active; Paper 1 alone preserves 504 free bits as 504 free bits.",
        "erasure_note":"Permutation-equivalent cubic coordinates provide alternate witnesses for most equations, but eight diagonal equations are unique and require explicit protection for arbitrary observation-coordinate erasure tolerance."
    }
}

out=ROOT/"results"/"cifar10_paper1_active_byte_codec.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2),flush=True)
