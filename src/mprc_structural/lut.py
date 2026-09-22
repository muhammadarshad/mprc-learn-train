from __future__ import annotations
from collections import Counter, defaultdict

def integer_evidence(counter: Counter, nclass: int):
    """One-vs-rest integer evidence: E_c = K*N_c - N."""
    if not counter:
        return None
    total = sum(counter.values())
    return [nclass * counter.get(c, 0) - total for c in range(nclass)]

class CategoricalLUT:
    def __init__(self, width: int, nclass: int):
        self.width = int(width)
        self.nclass = int(nclass)
        self.tables = [defaultdict(Counter) for _ in range(self.width)]

    def observe(self, row, label: int):
        for j, state in enumerate(row):
            self.tables[j][int(state)][int(label)] += 1

    def score(self, row):
        scores = [0] * self.nclass
        hits = 0
        for j, state in enumerate(row):
            ev = integer_evidence(self.tables[j].get(int(state)), self.nclass)
            if ev is None:
                continue
            hits += 1
            for c in range(self.nclass):
                scores[c] += ev[c]
        return scores, hits

    def predict(self, row):
        scores, hits = self.score(row)
        if hits == 0:
            return -1, scores, hits
        order = sorted(range(self.nclass), key=lambda c: scores[c], reverse=True)
        if scores[order[0]] == scores[order[1]]:
            return -1, scores, hits
        return order[0], scores, hits
