# v29 — Relational IDENTIFY Candidate

Status: precommitted candidate rule; no benchmark tuning.

The corrected directional ADI-9 remains nine Z256 bytes. QH4 is content/ring addressing and is not reinterpreted as a pixel-coordinate permutation.

For every aligned local query/candidate relation:

1. classify the nine ADI byte values through the integer QH4/J2 structural classifier when the cluster is addressable;
2. award one structural match when the J2 pattern types agree;
3. accumulate exact circular distance across the nine ADI bytes.

Across an observation, IDENTIFY returns the typed pair

    (number of structural pattern matches, total ADI circular energy).

Candidate selection is lexicographic:

    maximize structural matches,
    then minimize ADI energy.

All exact ties survive to downstream BIND -> REACT -> MEASURE.

No learned weight, threshold, Top-K value, float, or Softmax is introduced.

The rule is intentionally a candidate architecture choice. QH4/J2 and ADI exactness are established mathematics; their lexicographic composition is fixed here before any benchmark training so it can be falsified honestly.
