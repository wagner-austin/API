"""Growth-policy instrument across datasets of different shape.

Same three XGBoost arms as EXPERIMENT_2026-08-17_growth_policy_xgb_instrument
(depthwise d6 / lossguide L31 / lossguide L47), applied to two additional
datasets so the single-dataset caveat is answered rather than carried:
  - Taiwan bankruptcy: 6,819 x 95, ~3.2% positive (small, wide, very imbalanced)
  - German credit:     1,000 x 20, 30% positive   (tiny, mixed types, balanced-ish)
Stratified random 70/30 per seed (no grouping key exists in these datasets).
"""

from __future__ import annotations

import csv
import statistics as st
import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

EXT = r"C:\Users\Test\PROJECTS\API\services\covenant-radar-api\data\external"
N_EST, DEPTH, LR, MAX_BINS, MIN_LEAF = 200, 6, 0.05, 64, 20
SEEDS = (42, 43, 44)


def load_taiwan():
    with open(f"{EXT}\\kaggle_taiwan_bankruptcy\\data.csv", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    header, data = rows[0], rows[1:]
    y = np.array([int(r[0]) for r in data], dtype=np.int64)
    X = np.array([[float(v) for v in r[1:]] for r in data], dtype=np.float64)
    return X, y


def load_german():
    with open(f"{EXT}\\german_credit\\german.data", encoding="utf-8") as fh:
        rows = [line.split() for line in fh if line.strip()]
    y = np.array([1 if r[-1] == "2" else 0 for r in rows], dtype=np.int64)
    cols = list(zip(*[r[:-1] for r in rows]))
    feats = []
    for col in cols:
        try:
            feats.append([float(v) for v in col])
        except ValueError:
            codes = {v: i for i, v in enumerate(sorted(set(col)))}
            feats.append([float(codes[v]) for v in col])
    X = np.array(feats, dtype=np.float64).T
    return X, y


def mean_leaves(booster) -> float:
    df = booster.trees_to_dataframe()
    return float(df[df["Feature"] == "Leaf"].groupby("Tree").size().mean())


def run(name, X, y):
    import xgboost as xgb

    print(f"\n== {name}: {X.shape[0]} x {X.shape[1]}, positive {y.mean()*100:.2f}% ==")
    arms: dict[str, list[dict[str, float]]] = {}
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.30, random_state=seed, stratify=y
        )
        for arm, kw in [
            ("depthwise d6", {"grow_policy": "depthwise", "max_depth": DEPTH}),
            ("lossguide L31", {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 31}),
            ("lossguide L47", {"grow_policy": "lossguide", "max_depth": 0, "max_leaves": 47}),
        ]:
            t0 = time.perf_counter()
            m = xgb.XGBClassifier(
                n_estimators=N_EST, learning_rate=LR, max_bin=MAX_BINS,
                min_child_weight=MIN_LEAF, tree_method="hist",
                reg_alpha=0.0, reg_lambda=0.0, n_jobs=1, random_state=seed,
                eval_metric="logloss", **kw,
            ).fit(Xtr, ytr)
            t = time.perf_counter() - t0
            p = m.predict_proba(Xte)[:, 1]
            arms.setdefault(arm, []).append({
                "t": t,
                "auc": roc_auc_score(yte, p),
                "pr": average_precision_score(yte, p),
                "ll": log_loss(yte, p, labels=[0, 1]),
                "lv": mean_leaves(m.get_booster()),
            })
    print(f"{'arm':<15} {'fit s':>7} {'AUC-ROC':>8} {'AUC-PR':>8} {'log-loss':>9} {'leaves':>7}")
    for arm, rs in arms.items():
        print(f"{arm:<15} {st.mean(r['t'] for r in rs):>7.3f} "
              f"{st.mean(r['auc'] for r in rs):>8.4f} "
              f"{st.mean(r['pr'] for r in rs):>8.4f} "
              f"{st.mean(r['ll'] for r in rs):>9.4f} "
              f"{st.mean(r['lv'] for r in rs):>7.1f}")


def main() -> None:
    Xt, yt = load_taiwan()
    run("taiwan-bankruptcy", Xt, yt)
    Xg, yg = load_german()
    run("german-credit", Xg, yg)


if __name__ == "__main__":
    main()
