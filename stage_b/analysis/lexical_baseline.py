#!/usr/bin/env python3
"""Lexical baseline for the CC panel.

The question a reviewer asks first: does the 29-signal geometric panel beat a
bag-of-words model on the raw prompt text? If it does not, the panel is not
earning its keep, and the geometry may be reading surface properties of the
supplied candidate rather than anything about the model.

No model run. Prompts and labels are banked; stem_id groups the paired
prompts (same question, right vs hallucinated candidate), so splits are
grouped by stem to prevent a question appearing on both sides.
"""
import glob
import json
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

DATA = "/Users/msrk/Documents/commit-confluence/stage_b/data_bench"
SEED = 20260711
FOLDS = 5


def run(path):
    rows = [json.loads(l) for l in open(path)]
    X = [r["prompt"] for r in rows]
    y = np.array([r["label"] for r in rows])
    g = np.array([r["stem_id"] for r in rows])
    if len(set(y)) < 2:
        return None

    # Grouped by stem: the two members of a pair never straddle the split.
    gkf = GroupKFold(n_splits=FOLDS)
    oof = np.zeros(len(y), dtype=float)
    for tr, te in gkf.split(X, y, groups=g):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            max_features=50000),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        pipe.fit([X[i] for i in tr], y[tr])
        oof[te] = pipe.predict_proba([X[i] for i in te])[:, 1]
    return roc_auc_score(y, oof), len(y), len(set(g))


def main():
    print("Lexical baseline: TF-IDF (1-2gram) + logistic regression")
    print(f"Out-of-fold AUROC, GroupKFold(n={FOLDS}) grouped by stem_id\n")
    print(f"{'task':34s} {'n':>6s} {'stems':>6s} {'lexical AUROC':>14s}")
    print("-" * 64)
    out = {}
    for p in sorted(glob.glob(f"{DATA}/*.jsonl")):
        task = os.path.basename(p).split("_seed")[0]
        r = run(p)
        if r is None:
            continue
        auc, n, ns = r
        out[task] = auc
        print(f"{task:34s} {n:6d} {ns:6d} {auc:14.4f}")
    print()
    print("Compare against the geometric panel's reported AUROCs for the same")
    print("tasks. A lexical model that matches or beats the panel means the")
    print("label is recoverable from prompt surface form alone.")
    return out


if __name__ == "__main__":
    main()
