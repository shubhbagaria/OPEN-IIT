"""Phase 1 - ground truth.

1. Outcome label ("kept") from the payment ledger.
2. Six-way behavioural category: transparent rule-based hindsight labeller,
   checked against the human-annotated sample, then scaled to 100% of PTPs
   with confidence-gated self-training (pseudo-labelling).
"""
import numpy as np
import pandas as pd

from .config import (CATEGORIES, CAT2ID, CURE_DAYS, DATA_END, GRACE_DAYS, KEPT_THRESHOLD,
                     PARTIAL_THRESHOLD, ANNOTATOR_WEIGHT, SEED)


# =========================================================== 1. outcome label
def outcome_labels(D) -> pd.DataFrame:
    """Kept definition (agreed business rule):

        paid(t)  = sum of ALL-channel payments on the account in [captured_ts, t]
                   (a payment cannot be credited to two promises: a later PTP on the
                    same account supersedes the earlier one from its capture time)
        kept     = paid(promised_date + 2d grace, end of day) >= 90% * promised_amount
        cured    = paid(promised_date + 14d cure,  end of day) >= 90% * promised_amount
        status   = kept_on_time | kept_late (cure window) | partial (>=25%) | broken
    """
    p = D["ptps"][["ptp_id", "account_id", "captured_ts", "promised_date", "promised_amount"]].copy()
    p = p.sort_values(["account_id", "captured_ts"])
    p["next_capture"] = p.groupby("account_id")["captured_ts"].shift(-1)
    eod = pd.Timedelta(hours=23, minutes=59, seconds=59)
    p["grace_end"] = p["promised_date"] + pd.Timedelta(days=GRACE_DAYS) + eod
    p["cure_end"] = p["promised_date"] + pd.Timedelta(days=CURE_DAYS) + eod
    p["attrib_end_grace"] = p[["grace_end", "next_capture"]].min(axis=1)
    p["attrib_end_cure"] = p[["cure_end", "next_capture"]].min(axis=1)

    pay = D["payments"][["account_id", "payment_ts", "amount", "channel"]]
    m = p.merge(pay, on="account_id", how="left")
    m = m[(m["payment_ts"] >= m["captured_ts"]) & (m["payment_ts"] <= m["attrib_end_cure"])]
    m = m.sort_values(["ptp_id", "payment_ts"])
    m["cum"] = m.groupby("ptp_id")["amount"].cumsum()
    in_grace = m["payment_ts"] <= m["attrib_end_grace"]
    paid_grace = m[in_grace].groupby("ptp_id")["amount"].sum()
    paid_cure = m.groupby("ptp_id")["amount"].sum()
    hit = m[m["cum"] >= KEPT_THRESHOLD * m["promised_amount"]].groupby("ptp_id").first()
    settle_days = (hit["payment_ts"].dt.normalize() - hit["promised_date"]).dt.days
    first_pay = m.groupby("ptp_id")["payment_ts"].min()
    channels = m.groupby("ptp_id")["channel"].agg(lambda s: "|".join(sorted(set(s))))

    out = p[["ptp_id", "promised_amount", "cure_end", "grace_end", "next_capture", "promised_date"]].copy()
    out["paid_in_grace"] = out["ptp_id"].map(paid_grace).fillna(0.0)
    out["paid_in_cure"] = out["ptp_id"].map(paid_cure).fillna(0.0)
    out["pay_ratio_grace"] = out["paid_in_grace"] / out["promised_amount"]
    out["pay_ratio_cure"] = out["paid_in_cure"] / out["promised_amount"]
    out["kept"] = (out["pay_ratio_grace"] >= KEPT_THRESHOLD).astype(int)
    out["cured"] = (out["pay_ratio_cure"] >= KEPT_THRESHOLD).astype(int)
    out["settle_days"] = out["ptp_id"].map(settle_days)
    out["first_payment_ts"] = out["ptp_id"].map(first_pay)
    out["payment_channels"] = out["ptp_id"].map(channels).fillna("")
    out["superseded"] = (out["next_capture"].notna() & (out["next_capture"] < out["grace_end"])).astype(int)
    end = pd.Timestamp(DATA_END)
    out["grace_censored"] = (out["grace_end"] > end).astype(int)
    out["cure_censored"] = (out["cure_end"] > end).astype(int)
    out["status"] = np.select(
        [out["kept"] == 1, out["cured"] == 1, out["pay_ratio_cure"] >= PARTIAL_THRESHOLD],
        ["kept_on_time", "kept_late", "partial"], "broken")
    return out.drop(columns=["next_capture"]).reset_index(drop=True)


# =========================================================== 2a. rule-based hindsight labeller
def _nz(df, c):
    return df[c].fillna(0) if c in df else pd.Series(0, index=df.index)


def rule_labels(X: pd.DataFrame) -> pd.DataFrame:
    """Transparent, auditable decision list (hindsight: may use the payment outcome).

    X must contain end-of-call conversation features, account features and outcome
    columns. Returns rule label, a rule confidence in [0,1] and the fired rule id."""
    kept = _nz(X, "kept") > 0
    cured = _nz(X, "cured") > 0
    has_tx = X["n_turns"].notna()
    tp = (_nz(X, "n_c_third_party") + _nz(X, "n_c_relay") > 0) | (X["answerer_recorded"] != "self")
    cust_date = (_nz(X, "n_c_date") + _nz(X, "n_c_vague_date")) > 0
    no_commit = _nz(X, "recorded_without_commit") > 0
    no_cust_date = _nz(X, "recorded_without_cust_date") > 0
    pressure = _nz(X, "n_a_pressure")
    evasive = _nz(X, "n_c_exit") + _nz(X, "n_c_hedge") + _nz(X, "n_c_forgot") + _nz(X, "n_c_vague_date")
    firm = _nz(X, "n_c_firm") + _nz(X, "n_c_partial_offer") + _nz(X, "n_c_date")
    hardship = _nz(X, "n_c_hardship") > 0
    severe = _nz(X, "n_c_severe_hardship") > 0
    cond = _nz(X, "n_c_conditional") > 0
    prior_broken = _nz(X, "total_prior_broken")
    prior_ratio = X["total_broken_ratio"].fillna(0)
    low_atp = X["ability_to_pay_estimate"].fillna(0.55) < 0.45
    pre_sal = _nz(X, "is_pre_salary_hazard") > 0
    big_amt = X["amount_to_emi"].fillna(1) > 3

    partial_offer = _nz(X, "n_c_partial_offer") > 0
    prior_talk = (_nz(X, "n_a_prior_break") + _nz(X, "n_c_prior_break_excuse")) > 0
    exitish = (_nz(X, "n_c_exit") + _nz(X, "n_c_forgot")) > 0

    rules = [
        # (id, condition, label, confidence) - first match wins
        ("R01_third_party_with_date", has_tx & tp & cust_date, "third_party", 0.90),
        ("R02_third_party_logged_no_commit", has_tx & tp & ~cust_date, "agent_recorded_or_pushed", 0.70),
        ("R03_pushed_under_pressure", has_tx & no_cust_date & (pressure >= 1) & ~kept, "agent_recorded_or_pushed", 0.80),
        ("R04_logged_without_commit", has_tx & no_commit & no_cust_date & ~kept & (evasive == 0),
         "agent_recorded_or_pushed", 0.60),
        ("R05_kept_firm", kept & (firm > 0), "genuine_feasible", 0.90),
        ("R06_kept", kept, "genuine_feasible", 0.75),
        ("R07_cured_late_firm", cured & (firm > 0) & ~severe, "genuine_feasible", 0.60),
        ("R08_severe_hardship", severe, "genuine_infeasible", 0.90),
        ("R09_hardship_partial_or_conditional", hardship & (partial_offer | cond), "genuine_infeasible", 0.80),
        ("R10_chronic_breaker", (prior_broken >= 2) & (prior_talk | (prior_broken >= 3)) & (prior_ratio >= 0.5),
         "repeat_promiser", 0.80),
        ("R11_evasive_no_date", (evasive > 0) & no_cust_date, "escape", 0.80),
        ("R12_exit_after_promise", exitish & ~partial_offer, "escape", 0.65),
        ("R13_hardship_capacity", hardship & (pre_sal | low_atp | big_amt | cust_date), "genuine_infeasible", 0.65),
        ("R14_repeat_soft", (prior_broken >= 1) & prior_talk, "repeat_promiser", 0.60),
        ("R15_evasive", evasive > 0, "escape", 0.55),
        ("R16_hardship_or_partial", hardship | partial_offer | cured, "genuine_infeasible", 0.50),
        ("R17_no_tx_chronic", ~has_tx & (prior_broken >= 2), "repeat_promiser", 0.45),
        ("R18_no_tx_lowcap", ~has_tx & (low_atp | big_amt), "genuine_infeasible", 0.40),
        ("R19_no_tx_broken", ~has_tx, "escape", 0.35),
        ("R20_default", pd.Series(True, index=X.index), "genuine_feasible", 0.35),
    ]
    lab = pd.Series(None, index=X.index, dtype=object)
    conf = pd.Series(np.nan, index=X.index)
    rid = pd.Series(None, index=X.index, dtype=object)
    for r, cond_, l, c in rules:
        m = cond_.fillna(False).astype(bool) & lab.isna()
        lab[m], conf[m], rid[m] = l, c, r
    return pd.DataFrame({"rule_label": lab, "rule_conf": conf, "rule_id": rid}, index=X.index)


# =========================================================== 2b. semi-supervised bootstrapping
def _lgbm_multiclass(seed=SEED):
    import lightgbm as lgb
    return lgb.LGBMClassifier(objective="multiclass", num_class=len(CATEGORIES), n_estimators=250,
                              learning_rate=0.05, num_leaves=15, min_child_samples=15, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, random_state=seed,
                              verbose=-1)


def _fit_temperature(logp, y, w=None):
    """Temperature scaling on log-probabilities (1 parameter -> safe on 200 gold labels)."""
    from scipy.optimize import minimize_scalar
    w = np.ones(len(y)) if w is None else w

    def nll(t):
        z = logp / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -(w * lp[np.arange(len(y)), y]).sum() / w.sum()
    return minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded").x


def _apply_t(p, t):
    z = np.log(np.clip(p, 1e-9, 1)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def bootstrap_categories(X, feat_cols, gold, rounds=3, tau=0.85, seed=SEED, return_model=False):
    """Self-training from (gold annotations + high-precision rules) to 100% coverage.

    gold: DataFrame[ptp_id, annotated_type, annotator_confidence] - only the rows to train on.
    Round 0 seeds = gold (weight 3x annotator weight) + rule labels with rule_conf >= 0.75.
    Each round: fit -> temperature-calibrate on gold (out-of-fold) -> accept
    pseudo-labels whose calibrated confidence >= tau and that agree with the rule
    labeller (or exceed 0.95) -> refit. Returns per-PTP label, calibrated confidence, source."""
    from sklearn.model_selection import StratifiedKFold
    X = X.reset_index(drop=True)
    Xf = X[feat_cols]
    g = X[["ptp_id"]].merge(gold, on="ptp_id", how="left")
    is_gold = g["annotated_type"].notna().to_numpy()
    y_gold = g["annotated_type"].map(CAT2ID).to_numpy()
    w_gold = g["annotator_confidence"].map(ANNOTATOR_WEIGHT).fillna(0).to_numpy() * 3.0
    y_rule = X["rule_label"].map(CAT2ID).to_numpy()
    seed_rule = (~is_gold) & (X["rule_conf"].to_numpy() >= 0.75)

    y = np.where(is_gold, y_gold, np.where(seed_rule, y_rule, -1)).astype(float)
    w = np.where(is_gold, w_gold, np.where(seed_rule, 0.5 * X["rule_conf"].to_numpy(), 0.0))
    history = []
    T = 1.0
    for r in range(rounds + 1):
        m = y >= 0
        model = _lgbm_multiclass(seed).fit(Xf[m], y[m].astype(int), sample_weight=w[m])
        # out-of-fold gold predictions -> temperature
        gi = np.where(is_gold)[0]
        oof = np.zeros((len(gi), len(CATEGORIES)))
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for tr, te in skf.split(gi, y_gold[gi]):
            mm = m.copy()
            mm[gi[te]] = False
            mdl = _lgbm_multiclass(seed).fit(Xf[mm], y[mm].astype(int), sample_weight=w[mm])
            oof[te] = mdl.predict_proba(Xf.iloc[gi[te]])
        T = _fit_temperature(np.log(np.clip(oof, 1e-9, 1)), y_gold[gi].astype(int), w_gold[gi])
        oof_acc = (oof.argmax(1) == y_gold[gi]).mean()
        P = _apply_t(model.predict_proba(Xf), T)
        conf, pred = P.max(1), P.argmax(1)
        history.append(dict(round=r, n_train=int(m.sum()), temperature=round(T, 3), gold_oof_acc=round(oof_acc, 4)))
        if r == rounds:
            break
        accept = (~is_gold) & (conf >= tau) & ((pred == y_rule) | (conf >= 0.95))
        y = np.where(is_gold, y_gold, np.where(accept, pred, np.where(seed_rule, y_rule, -1))).astype(float)
        w = np.where(is_gold, w_gold, np.where(accept, conf, np.where(seed_rule, 0.5 * X["rule_conf"], 0.0)))

    label = np.where(is_gold, y_gold, pred).astype(int)
    out = pd.DataFrame({
        "ptp_id": X["ptp_id"],
        "category": [CATEGORIES[i] for i in label],
        "category_conf": np.where(is_gold, 1.0, conf),
        "category_source": np.where(is_gold, "human_annotation",
                                    np.where(conf >= tau, "pseudo_label_high_conf", "pseudo_label_low_conf")),
    })
    for i, c in enumerate(CATEGORIES):
        out["p_gt_" + c] = P[:, i]
    out.attrs["history"] = history
    if return_model:
        return out, model, T
    return out
