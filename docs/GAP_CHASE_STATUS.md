# Gap Chase Status

## Frozen basis

This research branch is constrained by the Sunday RC2 five-paper freeze and the Arshad's ViT v1 coding specification.

Do not change frozen constants to repair benchmark performance.

Key constraints used here:

- Z256, vacua {0,64,128,192}
- generator 7
- local ADI-9 = 1 reference + 8 relations
- native byte rectangles 16x7 <-> 7x16
- local Transpose T_k(a,p)=(p+k,a-k) mod 3
- Paper-3 order-two half-turn kernel and order-three faithfulness
- Papers 4/5 relation-code threshold d(K)>=5

## Best reproduced digit result

The fixed observation path is:

`u8 image -> deterministic integer channels -> QH4 quarter-pair relation state -> observation-populated class LUT -> integer score accumulation`

Selected channels from training/validation only:

- gy
- grad
- h2
- m4

Five fixed stratified splits:

| method | mean accuracy |
|---|---:|
| MPRC integer LUT | 95.81% |
| GD on exact same categorical QH4 observations | 97.07% |
| GD on richer raw deterministic channels | 98.07% |

Seed 42 integer LUT: 96.85%.

Controls:

- random-label: 3.33%
- independent per-image pixel shuffle: 10.74%

The model therefore learns spatial signal, but one-pixel translation robustness remains poor (~40-48%).

## Falsified gap hypotheses

These did not improve validation and are not promoted:

1. generator-7 phase-invariant pair/triple histograms;
2. quotienting the local T_k orbit;
3. T_k transport augmentation / orbit-summed inference;
4. mistake-driven integer perceptron correction;
5. training-only mutual-information pruning;
6. entropy/reliability gating;
7. alternative TP/TN/FP/FN scalar statistics;
8. small integer loudness weights across channel x relation-family groups.

The original count-posterior LUT repeatedly survives these controls.

## Exact T_k gate result

The three frozen T_k involutions generate six local permutations. On the 252 active ring states:

- all exact theorem gates pass;
- generated transport orbit size is 3 for every active state;
- quotienting gives 84 canonical active classes.

That quotient destroys useful classification information in the current digit representation, so T_k is transport/routing structure, not a classifier invariance to impose blindly.

## Next real-data gate

The Arshad-ViT research specification says to start real-data validation on CIFAR-10 and explicitly treats 32x32 as a stress test.

The current CI experiment:

- uses the official 50,000 / 10,000 split;
- does not resize;
- does not use 8x8 byte windows;
- uses explicit 7x16 and 16x7 rectangular structures;
- handles the 32-pixel non-closing remainder as explicit partial structures, not vacuum padding;
- compares horizontal phase, vertical phase, and both phases together.

This is a Layer-D transport diagnostic, not yet a claim that full B5 Arshad's ViT has been implemented.
