"""LightGBM vs XGBoost, like for like.

Re-uses the existing pipeline unchanged (same features, same stored ground-truth labels,
same account-level train/validation/test split, same calibration, same cost-sensitive
category decision, same evaluation functions). Only the gradient-boosting library differs.
Nothing in ptp_copilot/ is modified; results go to outputs/xgboost_comparison.json.

Run from the repository root:  python experiments/xgboost_comparison.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from ptp_copilot import pipeline as PL                      # noqa: E402
from ptp_copilot.config import CATEGORIES, OUT_DIR, SEED     # noqa: E402
from ptp_copilot.labels import _apply_t, _fit_temperature    # noqa: E402
from ptp_copilot.models import PortfolioCalibrator, ece      # noqa: E402
from ptp_copilot.config import COST_MATRIX                   # noqa: E402


# ----------------------------------------------------------------------------- XGBoost twins
# Settings mirror models.lgb_binary / CategoryModel: leaf-wise growth with 15 leaves,
# lr 0.02 / 0.03, subsample 0.8, colsample 0.7, L2 = 5, early stopping 100 on validation.
# LightGBM's min_child_samples (30 / 20 rows) is translated to a hessian-based
# min_child_weight (≈ rows × p(1−p)): 6 for the binary model, 3 for the multiclass model.
def xgb_params(**kw):
    p = dict(n_estimators=2000, learning_rate=0.02, tree_method="hist", grow_policy="lossguide", max_leaves=15,
             max_depth=0, min_child_weight=6, subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
             enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=100, random_state=SEED,
             n_jobs=-1, verbosity=0)
    p.update(kw)
    return p


class XGBKeep:
    """Same interface as models.KeepModel."""

    def __init__(self, name, features):
        self.name, self.features = name, list(features)

    def fit(self, Xtr, ytr, wtr, Xva, yva, wva=None, pf_va=None):
        t0 = time.time()
        self.model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **xgb_params()).fit(
            Xtr[self.features], ytr, sample_weight=wtr, eval_set=[(Xva[self.features], yva)],
            sample_weight_eval_set=[wva] if wva is not None else None, verbose=False)
        self.fit_s = time.time() - t0
        raw = self.raw(Xva)
        self.cal = PortfolioCalibrator().fit(raw, yva, pf_va)
        return self

    def raw(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X, portfolio):
        return self.cal.predict(self.raw(X), portfolio)


class XGBCategory:
    """Same interface as models.CategoryModel (balanced classes, temperature, cost-sensitive decision)."""

    def __init__(self, features):
        self.features = list(features)
        self.C = np.asarray(COST_MATRIX)

    def fit(self, Xtr, ytr, wtr, Xva, yva, wva=None):
        t0 = time.time()
        counts = np.bincount(ytr, minlength=len(CATEGORIES))
        balance = len(ytr) / (len(CATEGORIES) * np.maximum(counts, 1))      # == LightGBM class_weight="balanced"
        self.model = xgb.XGBClassifier(objective="multi:softprob", num_class=len(CATEGORIES), eval_metric="mlogloss",
                                       **xgb_params(learning_rate=0.03, min_child_weight=3)).fit(
            Xtr[self.features], ytr, sample_weight=wtr * balance[ytr], eval_set=[(Xva[self.features], yva)],
            sample_weight_eval_set=[wva] if wva is not None else None, verbose=False)
        self.fit_s = time.time() - t0
        P = self.model.predict_proba(Xva[self.features])
        self.T = _fit_temperature(np.log(np.clip(P, 1e-9, 1)), np.asarray(yva), wva)
        return self

    def proba(self, X):
        return _apply_t(self.model.predict_proba(X[self.features]), self.T)

    def decide(self, P):
        return (P @ self.C).argmin(1)


def train_xgb_stage_models(X, conv_cols, tr, va):
    """Mirror of pipeline.train_stage_models with XGBoost twins."""
    pre_cols = PL.ACC_FEATS + PL.AGENT_FEATS
    in_cols = pre_cols + PL.PTP_FEATS + conv_cols
    post_cols = in_cols + PL.POST_FEATS
    M = PL.model_frame(X, list(dict.fromkeys(post_cols)))
    y, w, pf_ = X["kept"].to_numpy(), X["w"].to_numpy(), X["portfolio"].to_numpy()
    fin = X["is_final"].to_numpy() == 1
    vf = va & fin
    m = {}
    m["pre_call"] = XGBKeep("pre_call", pre_cols).fit(M[tr & fin], y[tr & fin], None, M[vf], y[vf], None, pf_[vf])
    m["in_call"] = XGBKeep("in_call", in_cols).fit(M[tr], y[tr], w[tr], M[vf], y[vf], None, pf_[vf])
    m["post_call"] = XGBKeep("post_call", post_cols).fit(M[tr & fin], y[tr & fin], None, M[vf], y[vf], None, pf_[vf])
    cw = w * np.where(X["category_source"] == "human_annotation", 2.0,
                      np.where(X["category_source"] == "pseudo_label_high_conf", 1.0, 0.5))
    yc = X["y_cat"].to_numpy()
    m["category"] = XGBCategory(in_cols).fit(M[tr], yc[tr], cw[tr], M[vf], yc[vf], cw[vf])
    return m, M


# ----------------------------------------------------------------------------- helpers
def timed(fn):
    t0 = time.time()
    out = fn()
    return out, time.time() - t0


def crossfit(X, conv_cols, trainer, folds=5):
    """Same account-grouped folds as pipeline.crossfit_scores (same seed)."""
    acc_ids = X["account_id"].unique()
    rng = np.random.RandomState(SEED)
    fold_of = dict(zip(acc_ids, rng.randint(0, folds, len(acc_ids))))
    f = X["account_id"].map(fold_of).to_numpy()
    fin = X["is_final"].to_numpy() == 1
    rows = []
    for k in range(folds):
        test = f == k
        rest = [a for a in acc_ids if fold_of[a] != k]
        va_acc = set(rng.choice(rest, int(0.18 * len(rest)), replace=False))
        va = (~test) & X["account_id"].isin(va_acc).to_numpy()
        tr = (~test) & ~va
        models, M = trainer(X, conv_cols, tr, va)
        sel = test & fin
        pf_ = X["portfolio"].to_numpy()[sel]
        Pc = models["category"].proba(M[sel])
        rows.append(pd.DataFrame({
            "kept": X.loc[sel, "kept"].to_numpy(), "y_cat": X.loc[sel, "y_cat"].to_numpy(),
            "pre": models["pre_call"].predict(M[sel], pf_), "in": models["in_call"].predict(M[sel], pf_),
            "post": models["post_call"].predict(M[sel], pf_), "cat": models["category"].decide(Pc)}))
    d = pd.concat(rows)
    return dict(keep_auc_pre_call=round(roc_auc_score(d.kept, d.pre), 4),
                keep_auc_in_call=round(roc_auc_score(d.kept, d["in"]), 4),
                keep_auc_post_call=round(roc_auc_score(d.kept, d.post), 4),
                keep_ece_in_call=round(ece(d.kept, d["in"]), 4),
                category_acc=round(accuracy_score(d.y_cat, d.cat), 4),
                category_macro_f1=round(f1_score(d.y_cat, d.cat, average="macro"), 4))


def single_row_latency(model_obj, M, n=300):
    """Per-call inference time for ONE row, as in the live copilot (ms)."""
    rows = M.sample(n, random_state=SEED)
    ts = []
    for i in range(n):
        r = rows.iloc[[i]]
        t0 = time.perf_counter()
        model_obj(r)
        ts.append((time.perf_counter() - t0) * 1000)
    return dict(p50=round(float(np.percentile(ts, 50)), 3), p95=round(float(np.percentile(ts, 95)), 3))


def summarise(R, curve, trig):
    rep = R["category_vs_ground_truth"]["report"]
    out = {f"keep_{s}": R[f"keep_{s}"] for s in ["pre_call", "in_call", "post_call"]}
    out.update(
        category_cost_sensitive=dict(acc=round(rep["accuracy"], 4), macro_f1=round(rep["macro avg"]["f1-score"], 4),
                                     weighted_f1=round(rep["weighted avg"]["f1-score"], 4)),
        category_argmax=dict(acc=round(R["category_argmax_vs_ground_truth"]["report"]["accuracy"], 4),
                             macro_f1=round(R["category_argmax_vs_ground_truth"]["report"]["macro avg"]["f1-score"], 4)),
        category_per_class_f1={c: round(rep[c]["f1-score"], 3) for c in CATEGORIES},
        category_ovr_auc=R["category_ovr_auc_macro"], category_ece=R["category_ece_top1"],
        category_vs_human_test=R.get("category_vs_human_test"),
        hardship_to_aggressive_argmax=R["hardship_misclassified_as_aggressive_argmax"],
        hardship_to_aggressive_costsens=R["hardship_misclassified_as_aggressive_costsens"],
        streaming_curve=curve.to_dict("records"), heard_enough_trigger=trig)
    return out


# ----------------------------------------------------------------------------- main
def main():
    T0 = time.time()
    D, lab, ts, pf, base, conv_cols = PL.build()
    gt = pd.read_csv(OUT_DIR / "ground_truth_labels.csv")        # the exact labels the LightGBM models used
    X = PL.snapshot_frame(base, pf, ts, gt)
    split = X["split"].to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    res = {"setup": dict(train_ptps=int(X[tr].ptp_id.nunique()), val_ptps=int(X[va].ptp_id.nunique()),
                         test_ptps=int(X[te].ptp_id.nunique()), snapshots=len(X),
                         xgboost=xgb.__version__, params_binary=xgb_params(), note=__doc__.strip().splitlines()[0])}
    for name, trainer in [("lightgbm", PL.train_stage_models), ("xgboost", train_xgb_stage_models)]:
        print(f"\n=================== {name.upper()} ===================", flush=True)
        (models, M), fit_s = timed(lambda: trainer(X, conv_cols, tr, va))
        R, _, _ = PL.evaluate(models, X, M, te, D["annotations"])
        curve, trig = PL.streaming_eval(models, X, M, te)
        s = summarise(R, curve, trig)
        s["train_seconds_all_4_models"] = round(fit_s, 1)
        s["trees_used"] = {k: int(getattr(m.model, "best_iteration_", getattr(m.model, "best_iteration", -1)) or -1)
                           for k, m in models.items()}
        inc = models["in_call"]
        s["single_row_latency_ms"] = dict(
            keep_in_call=single_row_latency(lambda r: inc.predict(r, np.array([r["portfolio"].iloc[0]])), M),
            category=single_row_latency(lambda r: models["category"].decide(models["category"].proba(r)), M))
        (cf, cf_s) = timed(lambda: crossfit(X, conv_cols, trainer))
        s["crossfit_all_ptps"] = cf
        s["crossfit_seconds"] = round(cf_s, 1)
        res[name] = s
        print(json.dumps({k: v for k, v in s.items() if k not in ("streaming_curve",)}, indent=1, default=str))
    res["runtime_s"] = round(time.time() - T0, 1)
    with open(OUT_DIR / "xgboost_comparison.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\n[done] {res['runtime_s']}s -> outputs/xgboost_comparison.json")


if __name__ == "__main__":
    main()
