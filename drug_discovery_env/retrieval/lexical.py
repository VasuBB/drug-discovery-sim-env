from __future__ import annotations

import math
from collections import Counter


class BM25Lite:
    def __init__(self, corpus: list[str]) -> None:
        self.corpus = corpus
        self.tokens = [doc.lower().split() for doc in corpus]
        self.doc_freq: Counter[str] = Counter()
        self.doc_lens = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 1.0
        for doc in self.tokens:
            for term in set(doc):
                self.doc_freq[term] += 1

    def score(self, query: str, idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        q_terms = query.lower().split()
        doc = self.tokens[idx]
        counts = Counter(doc)
        score = 0.0
        n_docs = max(1, len(self.tokens))
        for term in q_terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            tf = counts.get(term, 0)
            denom = tf + k1 * (1 - b + b * (len(doc) / self.avg_len))
            if denom > 0:
                score += idf * ((tf * (k1 + 1)) / denom)
        return score
