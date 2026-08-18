"""Growth-policy experiment: what does leaf-wise buy on the bankruptcy workload?

XGBoost implements both growth policies, so it serves as the instrument:
  - depthwise  max_depth=6            (ClearGBM's current shape)
  - lossguide  max_leaves=31          (LightGBM's shape, matched to its default)
  - lossguide  max_leaves=47          (matched to ClearGBM's measured mean leaves)
LightGBM (leaf-wise, 31) and ClearGBM (depth-wise, 6) run as anchors.
Same company-disjoint splits, same seeds as the 2026-08-17 three-way run.
"""

from __future__ import annotations

import csv
import statistics as st
import time

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

CSV = "tests/data/american_bankruptcy.csv"
N_EST, DEPTH, LR, MAX_BINS, MIN_LEAF = 200, 6, 0.05, 64, 20
SEEDS = (42, 43, 44)
REPEATS, WARMUPS = 3, 1


def load():
    companies, labels, rows = [], [], []
    with open(CSV, encoding="utf-8-sig") as fh:
        for rec in csv.DictReader(fh):
            companies.append(rec["company_name"])
            labels.append(0 if rec["status_label"].strip() == "alive" else 1)
            rows.append([float(rec[f"X{i}"]) for i in range(1, 19)])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64), companies


def split(companies, seed):
    uniq = sorted(set(companies))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    cut = int(0.70 * len(uniq))
    train_c = {uniq[i] for i in perm[:cut]}
    idx = np.arange(len(companies))
    mask = np.array([c in train_c for c in companies])
    return idx[mask], idx[~mask]


def timed(fit):
    for _ in range(WARMUPS):
        fit()
    return st.median([(lambda t0: (fit(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(REPEATS)])


def mean_leaves_xgb(booster) -> float:
    df = booster.trees_to_dataframe()
    leaves = df[df["Feature"] == "Leaf"].groupby("Tree").size()
    return float(leaves.mean())


def main() -> None:
    import xgboost as xgb
    import lightgbm as lgb
    from cleargbm.ensemble import train_gradient_boosting, predict_proba
    from cleargbm.types import GradientBoostingConfig

    X, y, companies = load()
    print(f"dataset: {X.shape[0]} x {X.shape[1]}, positive {y.mean()*100:.2f}%\n")

    arms = {}

    def record(name, seed, t, p, yte, leaves=None):
        arms.setdefault(name, []).append({
            "t": t,
            "auc": roc_auc_score(yte, p),
            "pr": average_precision_score(yte, p),
            "ll": log_loss(yte, p, labels=[0, 1]),
            "leaves": leaves,
        })

    for seed in SEEDS:
        tr, te = split(companies, seed)
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        holder = {}

        def xgb_fit(policy, **kw):
            def f():
                holder["m"] = xgb.XGBClassifier(
                    n_estimators=N_EST, learning_rate=LR, max_bin=MAX_BINS,
                    min_child_weight=MIN_LEAF, tree_method="hist",
                    grow_policy=policy, reg_alpha=0.0, reg_lambda=0.0,
                    n_jobs=1, random_state=seed, eval_metric="logloss", **kw,
                ).fit(Xtr, ytr)
            return f

        for name, fit in [
            ("xgb depthwise d6", xgb_fit("depthwise", max_depth=DEPTH)),
            ("xgb lossguide L31", xgb_fit("lossguide", max_depth=0, max_leaves=31)),
            ("xgb lossguide L47", xgb_fit("lossguide", max_depth=0, max_leaves=47)),
        ]:
            t = timed(fit)
            m = holder["m"]
            p = m.predict_proba(Xte)[:, 1]
            record(name, seed, t, p, yte, mean_leaves_xgb(m.get_booster()))

        def lgb_fit():
            holder["l"] = lgb.LGBMClassifier(
                n_estimators=N_EST, max_depth=DEPTH, learning_rate=LR,
                max_bin=MAX_BINS, min_child_samples=MIN_LEAF, num_leaves=31,
                reg_alpha=0.0, reg_lambda=0.0, n_jobs=1, random_state=seed,
                verbose=-1,
            ).fit(Xtr, ytr)

        t = timed(lgb_fit)
        p = holder["l"].predict_proba(Xte)[:, 1]
        record("lgb leafwise L31", seed, t, p, yte, 31.0)

        cfg = GradientBoostingConfig(
            n_estimators=N_EST, max_depth=DEPTH, learning_rate=LR,
            min_samples_split=2, min_samples_leaf=MIN_LEAF, max_features=None,
            max_bins=MAX_BINS, subsample=1.0, random_state=seed,
            track_contributions=False, monotonic_constraints=None,
            reg_alpha=0.0, reg_lambda=0.0, n_jobs=1, early_stopping_rounds=None,
        )
        names = tuple(f"X{i}" for i in range(1, 19))

        def cg_fit():
            holder["c"] = train_gradient_boosting(Xtr, ytr, None, None, cfg, names)

        t = timed(cg_fit)
        p = predict_proba(holder["c"], Xte)[:, 1]
        record("cleargbm depthwise d6", seed, t, p, yte, None)

        print(f"  seed {seed} done")

    print(f"\n{'arm':<22} {'fit s':>8} {'AUC-ROC':>8} {'AUC-PR':>8} {'log-loss':>9} {'leaves':>7}")
    for name, rs in arms.items():
        lv = [r["leaves"] for r in rs if r["leaves"] is not None]
        print(f"{name:<22} {st.mean(r['t'] for r in rs):>8.3f} "
              f"{st.mean(r['auc'] for r in rs):>8.4f} "
              f"{st.mean(r['pr'] for r in rs):>8.4f} "
              f"{st.mean(r['ll'] for r in rs):>9.4f} "
              f"{(f'{st.mean(lv):.1f}' if lv else '  n/a'):>7}")


if __name__ == "__main__":
    main()
