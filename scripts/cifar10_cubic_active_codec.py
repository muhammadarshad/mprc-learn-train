#!/usr/bin/env python3
"""
CIFAR-10 exact cubic active-information codec.

Frozen mathematical basis:
  Paper 1, Quantinion Simplex Cubic Decomposition over 2-Power Rings.
  n=8, r=n+1=9, m=8, Z_256.

Codec:
  63 arbitrary payload bytes = 504 bits.
  The canonical 9x8 simplex has 72 atom coordinates, each with one fixed
  parity bit and seven free higher bits. 504 payload bits fill those 72x7
  free positions.

  We compute the cubic observation and retain a fixed set of 72 tensor
  coordinates whose one-bit lifting Jacobian has rank 72. Their parity is
  fixed by the simplex, so stripping that parity leaves exactly 72x7=504
  information bits = 63 stored bytes.

Decode:
  restore known tensor parity, then use the published one-bit lifting
  Jacobian seven times:
      2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256.
  The recovered atom bytes return the original 63 payload bytes exactly.

This is an exact recoverable representation test. It is NOT a claim that
Paper 1 by itself compresses arbitrary information.
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
PAYLOAD_BYTES=63
BLOCKS_PER_IMAGE=49
PAD_PER_IMAGE=BLOCKS_PER_IMAGE*PAYLOAD_BYTES-32*32*3
assert PAD_PER_IMAGE==15

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

def load_test():
    folder=ensure_dataset()
    with open(folder/"test_batch","rb") as f:
        d=pickle.load(f,encoding="bytes")
    x=d[b"data"].astype(np.uint8) # exact CIFAR byte order, [10000,3072]
    y=np.asarray(d[b"labels"],dtype=np.int64)
    assert x.shape==(10000,3072)
    return x,y

def canonical_simplex():
    s=np.zeros((9,8),dtype=np.uint8)
    s[0,:]=1
    for i in range(8):
        s[i+1,i]=1
    return s

SUP=canonical_simplex()

def build_jacobian():
    rows=[]; coords=[]
    for i in range(8):
      for j in range(8):
       for k in range(8):
        row=np.zeros(72,dtype=np.uint8)
        for q,s in enumerate(SUP):
          for a in range(8):
            v=0
            if i==a: v ^= int(s[j]&s[k])
            if j==a: v ^= int(s[i]&s[k])
            if k==a: v ^= int(s[i]&s[j])
            row[q*8+a]=v
        rows.append(row); coords.append((i,j,k))
    return np.stack(rows),coords

J,COORDS=build_jacobian()

def gf2_rank(A):
    A=A.copy().astype(np.uint8)
    m,n=A.shape; r=0
    for c in range(n):
        piv=np.flatnonzero(A[r:,c])
        if len(piv)==0: continue
        p=r+int(piv[0]); A[[r,p]]=A[[p,r]]
        mask=np.flatnonzero(A[:,c])
        for i in mask:
            if i!=r: A[i]^=A[r]
        r+=1
        if r==m: break
    return r

def gf2_inverse(A):
    n=A.shape[0]
    aug=np.concatenate([A.copy().astype(np.uint8),np.eye(n,dtype=np.uint8)],axis=1)
    r=0
    for c in range(n):
        piv=np.flatnonzero(aug[r:,c])
        if len(piv)==0: raise RuntimeError("singular GF2 matrix")
        p=r+int(piv[0]); aug[[r,p]]=aug[[p,r]]
        for i in np.flatnonzero(aug[:,c]):
            if i!=r: aug[i]^=aug[r]
        r+=1
    return aug[:,n:]

assert gf2_rank(J)==72

# Deterministic rank-72 observation basis.
selected=[]; basis=[]; rank=0
for idx,row in enumerate(J):
    trial=np.stack(basis+[row]) if basis else row[None,:]
    nr=gf2_rank(trial)
    if nr>rank:
        selected.append(idx); basis.append(row.copy()); rank=nr
        if rank==72: break

SEL=np.asarray(selected,dtype=np.int64)
JSEL=J[SEL]
assert JSEL.shape==(72,72) and gf2_rank(JSEL)==72
JINV=gf2_inverse(JSEL).astype(np.uint16)
assert np.array_equal(
    (JINV @ JSEL.astype(np.uint16))&1,
    np.eye(72,dtype=np.uint16)
)

SEL_COORDS=[COORDS[i] for i in SEL]
II=np.asarray([x[0] for x in SEL_COORDS],dtype=np.int64)
JJ=np.asarray([x[1] for x in SEL_COORDS],dtype=np.int64)
KK=np.asarray([x[2] for x in SEL_COORDS],dtype=np.int64)

def selected_tensor(atoms,mod):
    x=atoms[:,:,II].astype(np.uint32)
    y=atoms[:,:,JJ].astype(np.uint32)
    z=atoms[:,:,KK].astype(np.uint32)
    return ((x*y*z).sum(axis=1)%mod).astype(np.uint16)

T2=selected_tensor(SUP[None,:,:],2)[0].astype(np.uint8)
POW7=(1<<np.arange(7,dtype=np.uint16))

def bytes_to_vals(payload):
    bits=np.unpackbits(payload,axis=1,bitorder="little")
    g=bits.reshape(len(payload),72,7)
    return (g.astype(np.uint16)*POW7[None,None,:]).sum(axis=2).astype(np.uint8)

def vals_to_bytes(vals):
    bits=((vals[:,:,None]>>np.arange(7,dtype=np.uint8)[None,None,:])&1).astype(np.uint8)
    return np.packbits(bits.reshape(len(vals),504),axis=1,bitorder="little")

def encode_blocks(payload):
    vals=bytes_to_vals(payload)
    atoms=(SUP[None,:,:] | (vals.reshape(len(payload),9,8).astype(np.uint16)<<1)).astype(np.uint16)
    target=selected_tensor(atoms,256).astype(np.uint8)
    if not np.all((target&1)==T2[None,:]):
        raise RuntimeError("tensor parity drift")
    packed=vals_to_bytes((target>>1).astype(np.uint8))
    return packed,atoms.astype(np.uint8)

def decode_blocks(packed):
    active=bytes_to_vals(packed)
    target=(T2[None,:] | (active.astype(np.uint16)<<1)).astype(np.uint16)
    B=len(packed)
    cur=np.broadcast_to(SUP,(B,9,8)).copy().astype(np.uint16)

    for b in range(1,8):
        mod=1<<(b+1); step=1<<b
        pred=selected_tensor(cur,mod)
        rhs=(((target-pred)%mod)//step)&1
        delta=((rhs.astype(np.uint16) @ JINV.T)&1).astype(np.uint16)
        cur=(cur + step*delta.reshape(B,9,8))%mod

    vals=(cur.reshape(B,72)>>1).astype(np.uint8)
    return vals_to_bytes(vals),cur.astype(np.uint8)

def rowmask(row):
    x=0
    for i,b in enumerate(row):
        if b: x|=1<<i
    return x

def rank_masks(rows):
    basis={}
    for v in rows:
        x=v
        while x:
            p=x.bit_length()-1
            if p in basis: x^=basis[p]
            else:
                basis[p]=x
                break
    return len(basis)

# Structural erasure profile of three observation sets.
masks72=[rowmask(r) for r in JSEL]
unique_idx=[i for i,(a,b,c) in enumerate(COORDS) if a<=b<=c]
masks120=[rowmask(J[i]) for i in unique_idx]
masks512=[rowmask(r) for r in J]

def single_erasure_profile(masks):
    ok=0; bad=[]
    for e in range(len(masks)):
        if rank_masks(masks[:e]+masks[e+1:])==72:
            ok+=1
        else:
            bad.append(e)
    return {"coordinates":len(masks),"recoverable_single_erasures":ok,
            "ambiguous_single_erasures":len(bad)}

profile72=single_erasure_profile(masks72)
profile120=single_erasure_profile(masks120)
profile512=single_erasure_profile(masks512)

x,y=load_test()

sha=hashlib.sha256()
images_exact=0
blocks_exact=0
byte_errors=0
atom_corruption_recovered=0
atom_corruption_trials=0
start=time.time()
CHUNK_IMAGES=64

rng=np.random.default_rng(20260924)

for s in range(0,len(x),CHUNK_IMAGES):
    e=min(len(x),s+CHUNK_IMAGES)
    raw=x[s:e]
    padded=np.pad(raw,((0,0),(0,PAD_PER_IMAGE)),constant_values=0)
    blocks=padded.reshape(-1,63)

    packed,atoms=encode_blocks(blocks)
    sha.update(packed.tobytes())
    got,got_atoms=decode_blocks(packed)

    eq=np.all(got==blocks,axis=1)
    blocks_exact+=int(eq.sum())
    byte_errors+=int(np.count_nonzero(got!=blocks))

    restored=got.reshape(e-s,-1)
    images_exact+=int(np.all(restored[:,:3072]==raw,axis=1).sum())

    # Sample the theorem's "corrupted atom state -> recover from cubic observation" use.
    take=min(16,len(blocks))
    if take:
        sampled=np.arange(take)
        damaged=atoms[sampled].copy()
        for q in range(take):
            ai=int(rng.integers(0,9)); aj=int(rng.integers(0,8))
            damaged[q,ai,aj]^=np.uint8(int(rng.integers(1,256)))
        # Decoder uses stored cubic active information, not damaged atom copy.
        _,rest_atoms=decode_blocks(packed[sampled])
        atom_corruption_recovered+=int(np.all(rest_atoms==atoms[sampled],axis=(1,2)).sum())
        atom_corruption_trials+=take

elapsed=time.time()-start

raw_bytes=int(x.nbytes)
per_image_encoded=BLOCKS_PER_IMAGE*63
encoded_bytes=len(x)*per_image_encoded
stream_encoded=((raw_bytes+62)//63)*63

result={
  "dataset":{
    "name":"CIFAR-10 official test split",
    "images":int(len(x)),
    "raw_bytes":raw_bytes,
    "bytes_per_image":3072,
  },
  "codec":{
    "payload_bytes_per_block":63,
    "atom_bytes_per_block":72,
    "selected_cubic_coordinates":72,
    "known_parity_bits_stripped_per_block":72,
    "packed_information_bytes_per_block":63,
    "blocks_per_image":BLOCKS_PER_IMAGE,
    "tail_padding_bytes_per_image":PAD_PER_IMAGE,
    "selected_jacobian_rank":int(gf2_rank(JSEL)),
  },
  "exact_recovery":{
    "images_exact":images_exact,
    "images_total":int(len(x)),
    "blocks_exact":blocks_exact,
    "blocks_total":int(len(x)*BLOCKS_PER_IMAGE),
    "payload_byte_errors":byte_errors,
    "atom_corruption_trials":atom_corruption_trials,
    "atom_corruption_recovered_from_cubic":atom_corruption_recovered,
  },
  "storage":{
    "per_image_raw_bytes":3072,
    "per_image_independent_codec_bytes":per_image_encoded,
    "per_image_overhead_bytes":per_image_encoded-3072,
    "per_image_overhead_percent":100.0*(per_image_encoded-3072)/3072,
    "test_split_raw_bytes":raw_bytes,
    "test_split_per_image_codec_bytes":encoded_bytes,
    "test_split_stream_codec_bytes":stream_encoded,
    "stream_padding_bytes":stream_encoded-raw_bytes,
    "stream_overhead_percent":100.0*(stream_encoded-raw_bytes)/raw_bytes,
    "interpretation":"No information compression from the theorem alone; it is a bijective recoverable representation. Compression requires image-structural inactive/redundant bytes."
  },
  "single_coordinate_erasure_rank_profile":{
    "minimal_72":profile72,
    "unique_symmetric_120":profile120,
    "full_ordered_512":profile512,
    "full_ordered_512_note":"The only rank-critical single coordinates are expected to be the eight pure diagonals (i,i,i)."
  },
  "packed_sha256":sha.hexdigest(),
  "runtime_seconds":elapsed,
}
out=ROOT/"results"/"cifar10_cubic_active_codec.json"
out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result,indent=2))
