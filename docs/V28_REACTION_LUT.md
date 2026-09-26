# v28 — Observation-Populated ReactionLUT Candidate

Status: exact optimizer for a stated local objective; candidate architecture rule.

No benchmark result is used to choose this rule.

For a five-site staple input u in Z256, collect observed center bytes c into counts H[u,c]. Define

L[u] in argmin over y in Z256 of sum_c H[u,c] * d256(y,c).

The deterministic tie break is ring-relative.

This gives a complete 256-byte ReactionLUT.

Because each L[u] appears in only one summand of the total reconstruction objective, the 256 minimizations are independent. Exhaustive search over the 256 possible outputs is therefore the exact global optimizer for the stated objective.

Unseen u has zero objective for every output; the neutral convention is L[u]=u.

The rule is byte-native, integer-only, label-free, observation-populated, gradient-free, Softmax-free, and exactly 256 outputs.

Claim boundary: the optimizer theorem is exact conditional on the stated reconstruction objective. The choice of local center reconstruction as the learning objective is a candidate architecture rule, not a previously frozen MPRC theorem.
