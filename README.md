# MPRC Structural Learning

Standalone empirical research repository for observation-populated MPRC learning.

## What is learned

- `W^(2)` is an observed order-2 relation `(x,y) in Z_256^2`.
- `B^(1)=Delta(W)=x-y mod 256` is a derived order-1 differential.
- `Sigma(W)=x+y mod 256` is a derived accumulation.
- LUT occupancy/evidence is populated from observations; `W`, `B`, and `Sigma` are not free scalar parameters tuned by GD.

## Empirical dataset

The exact 1,797-sample handwritten-digit dataset used by the experiments is vendored at `data/digits.csv`, with the exact stratified 1,257/540 split at `data/split_seed42.csv`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/reproduce_all.sh
```

## Experiments

### W²/B¹ exhaustive ring test

`python scripts/test_W2_B1.py`

Checks all 65,536 ordered Z256 pairs.

### Real handwritten digits

`python scripts/train_digits.py`

Pipeline:

`image -> LoG -> Z256 -> local + stride-7 W² relations -> LUT occupancy/evidence -> integer class decision`

Includes logistic-regression controls on the same split.

### Arshad's ViT + VOF ablation

`python scripts/vit_vof_ablation.py`

Tests:
- local ADI-9
- generator-7 walk
- VOF-derived integer second-difference curvature primitive
- S5 pre-LUT REACT observable

The VOF transfer is intentionally narrow: only the integer second-difference primitive is used. The full PLIC/VOF solver is **not** claimed to be a vision classifier.

## Current recorded results

- MPRC local + stride-7 W²/B¹ evidence: ~83.89% held-out accuracy.
- ADI-9 + walk-7 + VOF second-difference ablation: ~84.63%.
- GD logistic control on numeric Z256: ~96.48%.
- GD logistic control on float LoG: ~96.67%.

These results do **not** claim MPRC beats GD. They show empirical learnability of observation-populated discrete relation LUTs and quantify the current quality gap.

## Source specifications

`docs/Arshads_ViT_Coding_Spec_v1.docx`

`docs/QH4_VOF_Interface_Reconstruction.docx`

These are treated as source specifications. Experiments keep ViT and VOF roles distinct.

## Provenance

This repository is the authoritative upstream research home. A downstream snapshot/subtree may later be vendored into `muhammadarshad/mprc-fft`.
