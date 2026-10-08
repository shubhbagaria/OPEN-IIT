"""Phase 3 - tabular credibility models (LightGBM) + calibration + SHAP reason codes.

Three time-staged keep models share one design:
  pre_call   : account + history + agent integrity                     (dial planning)
  in_call    : pre_call + conversation prefix + acoustics + PTP terms  (live, every customer turn)
  post_call  : in_call(final) + agent-entered fields + <=36h engagement (parking decision)
plus a 6-way category model on the in-call feature set with cost-sensitive decoding.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from .config import CATEGORIES, COST_MATRIX, SEED


# ----------------------------------------------------------------- metrics
def ece(y, p, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def multiclass_ece(y, P, n_bins=10):
    conf, pred = P.max(1), P.argmax(1)
    return ece((pred == y).astype(int), conf, n_bins)


# ----------------------------------------------------------------- binary keep model
def lgb_binary(**kw):
    params = dict(objective="binary", n_estimators=2000, learning_rate=0.02, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=5.0, random_state=SEED, verbose=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


class PortfolioCalibrator:
    """Global Platt scaling + per-portfolio intercept offset (empirical-Bayes shrunk).

    Within a portfolio the mapping is strictly monotone, so ranking (AUC) is
    preserved; across portfolios the offset aligns mean predicted and observed
    keep rates so that a score of 0.6 means the same thing for a salaried
    personal-loan borrower and a microfinance borrower. Two global parameters +
    one shrunk offset per portfolio is deliberately low-variance: the
    validation set has only ~560 PTPs."""

    def __init__(self, shrink_n=150):
        self.shrink_n = shrink_n

    def fit(self, raw, y, portfolio):
        raw, y, portfolio = np.asarray(raw), np.asarray(y), np.asarray(portfolio)
        z = _logit(raw).reshape(-1, 1)
        lr = LogisticRegression(C=10.0).fit(z, y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        self.offset = {}
        g = self.a * z[:, 0] + self.b
        for pf in pd.unique(portfolio):
            m = portfolio == pf
            # one Newton step for the intercept offset, then shrink toward 0
            p = 1 / (1 + np.exp(-g[m]))
            step = (y[m] - p).sum() / max((p * (1 - p)).sum(), 1e-6)
            self.offset[pf] = float(step * m.sum() / (m.sum() + self.shrink_n))
        return self

    def predict(self, raw, portfolio):
        g = self.a * _logit(np.asarray(raw)) + self.b
        off = np.array([self.offset.get(p, 0.0) for p in np.asarray(portfolio)])
        return np.clip(1 / (1 + np.exp(-(g + off))), 0.01, 0.99)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class KeepModel:
    def __init__(self, name, features):
        self.name, self.features = name, list(features)

    def fit(self, Xtr, ytr, wtr, Xva, yva, wva=None, pf_va=None):
        self.model = lgb_binary().fit(
            Xtr[self.features], ytr, sample_weight=wtr, eval_set=[(Xva[self.features], yva)],
            eval_sample_weight=[wva] if wva is not None else None,
            callbacks=[lgb.early_stopping(100, verbose=False)])
        raw = self.model.predict_proba(Xva[self.features])[:, 1]
        self.cal = PortfolioCalibrator().fit(raw, yva, pf_va)
        return self

    def raw(self, X):
        return self.model.predict_proba(X[self.features])[:, 1]

    def predict(self, X, portfolio):
        return self.cal.predict(self.raw(X), portfolio)

    def shap(self, X):
        """Exact TreeSHAP contributions (log-odds space) via LightGBM's native pred_contrib."""
        c = self.model.booster_.predict(X[self.features], pred_contrib=True)
        return pd.DataFrame(c[:, :-1], columns=self.features, index=X.index)


# ----------------------------------------------------------------- category model
class CategoryModel:
    def __init__(self, features):
        self.features = list(features)
        self.C = np.asarray(COST_MATRIX)

    def fit(self, Xtr, ytr, wtr, Xva, yva, wva=None):
        self.model = lgb.LGBMClassifier(
            objective="multiclass", num_class=len(CATEGORIES), n_estimators=2000, learning_rate=0.03,
            num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=5.0, class_weight="balanced", random_state=SEED, verbose=-1,
        ).fit(Xtr[self.features], ytr, sample_weight=wtr, eval_set=[(Xva[self.features], yva)],
              eval_sample_weight=[wva] if wva is not None else None,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        from .labels import _fit_temperature
        P = self.model.predict_proba(Xva[self.features])
        self.T = _fit_temperature(np.log(np.clip(P, 1e-9, 1)), np.asarray(yva), wva)
        return self

    def proba(self, X):
        from .labels import _apply_t
        return _apply_t(self.model.predict_proba(X[self.features]), self.T)

    def decide(self, P):
        """Bayes decision under the asymmetric cost matrix: argmin_j sum_i P_i * C[i, j]."""
        return (P @ self.C).argmin(1)
