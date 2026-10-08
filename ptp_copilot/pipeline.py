"""End-to-end offline pipeline: data -> labels -> features -> models -> decisions -> audits.

Run:  python -m ptp_copilot.pipeline
"""
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, brier_score_loss)

from . import features as F
from . import labels as L
from . import policy as P
from .config import CATEGORIES, CAT2ID, OUT_DIR, MODEL_DIR, SEED, COST
from .data import load_all
from .models import CategoryModel, KeepModel, ece, multiclass_ece

warnings.filterwarnings("ignore")
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


ACC_FEATS = ["lender_id", "portfolio", "income_type", "preferred_language", "bucket_ord", "dpd_start", "emi_amount",
             "overdue_start", "outstanding", "salary_credit_day", "bureau_ord", "other_active_loans",
             "paid_other_lenders_30d", "last_bounce_reason", "ability_to_pay_estimate", "prev_ptp_count",
             "prev_ptp_broken", "prev_broken_ratio", "overdue_to_outstanding", "overdue_to_emi", "salary_known",
             "hist_ptps_any", "hist_ptps_closed", "hist_kept", "hist_broken", "hist_settle_days", "total_prior_ptps",
             "total_prior_broken", "total_broken_ratio", "days_since_prev_ptp", "dial_30d", "answered_30d",
             "refused_30d", "tp_contact_30d", "hardship_30d", "answer_rate_30d", "pay_cnt_30d", "pay_amt_30d"]
AGENT_FEATS = ["agent_keep_rate", "agent_keep_lift", "agent_shift_end_share", "agent_generic_remark_share",
               "agent_ptps_per_day"]
PTP_FEATS = ["channel", "promised_amount", "overdue_at_capture", "promise_lag_days", "amount_to_overdue",
             "amount_to_emi", "amount_x_atp", "promised_dow", "promised_dom", "capture_hour", "capture_dow",
             "days_after_salary", "is_salary_window", "is_pre_salary_hazard", "late_shift_capture"]
POST_FEATS = ["answerer_recorded", "answerer_not_self", "payment_link_sent", "remark_generic", "remark_copy_count",
              "capture_disposition", "pc_payment_link_opened", "pc_confirmation_reply_yes",
              "pc_confirmation_reply_no", "pc_inbound_call_from_customer", "pc_whatsapp_confirmation_sent",
              "pc_hours_to_link_open"]
NON_FEATURE_PREFIX = {"call_id", "turn_idx", "first_commit_turn", "cust_last_day", "day_under_discussion"}


# =========================================================================== build
def build():
    t0 = time.time()
    D = load_all()
    lab = L.outcome_labels(D)
    ts = F.turn_signals(D["transcripts"])
    pf = F.prefix_features(ts)
    conv_cols = [c for c in pf.columns if c not in NON_FEATURE_PREFIX] + ["discussed_day_vs_salary"]
    acc = F.account_features(D, lab)
    acc["late_shift_capture"] = (acc["captured_ts"].dt.hour >= 17).astype(int)
    fit_mask = (acc["split"] == "train").to_numpy()
    ag = F.agent_integrity(acc, lab, fit_mask)
    pc = F.postcall_features(D)
    base = acc.merge(ag, on="ptp_id").merge(pc, on="ptp_id", how="left") \
              .merge(lab.drop(columns=["promised_amount", "promised_date"]), on="ptp_id")
    log(f"[build] features in {time.time() - t0:.1f}s  ptps={len(base)} turns={len(ts)}")
    return D, lab, ts, pf, base, conv_cols


def add_conv(base, conv_frame):
    x = base.merge(conv_frame, left_on="attempt_id", right_on="call_id", how="left")
    sal = x["salary_day_said"].fillna(x["salary_credit_day"])
    x["discussed_day_vs_salary"] = F._days_from_salary(x["day_under_discussion"].to_numpy(float), sal.to_numpy(float))
    return x


# =========================================================================== ground truth categories
def ground_truth(D, base, pf):
    X = add_conv(base, F.call_level(pf).reset_index())
    X = X.join(L.rule_labels(X))
    drop = {"ptp_id", "account_id", "attempt_id", "visit_id", "agent_id", "captured_ts", "promised_date", "split",
            "remark", "call_id", "status", "rule_label", "rule_conf", "rule_id", "town_id", "payment_channels"}
    feat = [c for c in X.columns if c not in drop and not pd.api.types.is_datetime64_any_dtype(X[c])]
    Xm = pd.concat([X[["ptp_id", "rule_label", "rule_conf"]], F.to_model_frame(X, feat)], axis=1)
    gt = L.bootstrap_categories(Xm, feat, D["annotations"], rounds=3)
    log("[labels] self-training history:", gt.attrs["history"])
    g = D["annotations"].merge(X[["ptp_id", "rule_label"]], on="ptp_id")
    log(f"[labels] rule labeller vs human (in-sample, rules were designed with this sample): "
        f"acc={accuracy_score(g.annotated_type, g.rule_label):.3f} "
        f"macroF1={f1_score(g.annotated_type, g.rule_label, average='macro'):.3f}")
    return gt.merge(X[["ptp_id", "rule_label", "rule_conf", "rule_id"]], on="ptp_id")


def gold_cv(D, base, pf, folds=5):
    """Honest estimate of the label pipeline: hold out annotation folds entirely."""
    from sklearn.model_selection import StratifiedKFold
    X = add_conv(base, F.call_level(pf).reset_index())
    X = X.join(L.rule_labels(X))
    drop = {"ptp_id", "account_id", "attempt_id", "visit_id", "agent_id", "captured_ts", "promised_date", "split",
            "remark", "call_id", "status", "rule_label", "rule_conf", "rule_id", "town_id", "payment_channels"}
    feat = [c for c in X.columns if c not in drop and not pd.api.types.is_datetime64_any_dtype(X[c])]
    Xm = pd.concat([X[["ptp_id", "rule_label", "rule_conf"]], F.to_model_frame(X, feat)], axis=1)
    gold = D["annotations"]
    res = []
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(gold, gold.annotated_type):
        out = L.bootstrap_categories(Xm, feat, gold.iloc[tr], rounds=2)
        res.append(gold.iloc[te].merge(out, on="ptp_id").merge(X[["ptp_id", "rule_label"]], on="ptp_id"))
    h = pd.concat(res)
    r = {}
    for col, name in [("rule_label", "rules_only"), ("category", "rules+self_training")]:
        r[name] = dict(acc=round(accuracy_score(h.annotated_type, h[col]), 3),
                       macro_f1=round(f1_score(h.annotated_type, h[col], average="macro"), 3))
    log("[labels] held-out gold agreement (5-fold):", r)
    return r, h


# =========================================================================== training frames
def snapshot_frame(base, pf, ts, gt):
    snaps = F.snapshot_rows(pf, ts)
    calls = add_conv(base, snaps)
    calls = calls[calls["call_id"].notna()]
    nofield = base[~base["attempt_id"].isin(snaps["call_id"])]           # field visits / missing transcript
    field = add_conv(nofield, snaps.iloc[0:0])
    field["is_final"] = 1
    X = pd.concat([calls, field], ignore_index=True)
    n = X.groupby("ptp_id")["ptp_id"].transform("size")
    X["w"] = np.where(n == 1, 1.0, np.where(X["is_final"] == 1, 0.5, 0.5 / (n - 1).clip(lower=1)))
    X = X.merge(gt[["ptp_id", "category", "category_source", "category_conf"]], on="ptp_id")
    X["y_cat"] = X["category"].map(CAT2ID)
    return X


def model_frame(X, cols):
    return F.to_model_frame(X, cols)


# =========================================================================== train / evaluate
def train_stage_models(X, conv_cols, train_mask, val_mask):
    pre_cols = ACC_FEATS + AGENT_FEATS
    in_cols = pre_cols + PTP_FEATS + conv_cols
    post_cols = in_cols + POST_FEATS
    all_cols = list(dict.fromkeys(post_cols))
    M = model_frame(X, all_cols)
    y, w, pf_ = X["kept"].to_numpy(), X["w"].to_numpy(), X["portfolio"].to_numpy()
    fin = X["is_final"].to_numpy() == 1
    tr, va = train_mask, val_mask
    vf = va & fin
    models = {}
    models["pre_call"] = KeepModel("pre_call", pre_cols).fit(M[tr & fin], y[tr & fin], None, M[vf], y[vf], None, pf_[vf])
    models["in_call"] = KeepModel("in_call", in_cols).fit(M[tr], y[tr], w[tr], M[vf], y[vf], None, pf_[vf])
    models["post_call"] = KeepModel("post_call", post_cols).fit(M[tr & fin], y[tr & fin], None, M[vf], y[vf], None, pf_[vf])
    # category: train on all snapshots, weight down low-confidence pseudo labels
    cw = w * np.where(X["category_source"] == "human_annotation", 2.0,
                      np.where(X["category_source"] == "pseudo_label_high_conf", 1.0, 0.5))
    yc = X["y_cat"].to_numpy()
    models["category"] = CategoryModel(in_cols).fit(M[tr], yc[tr], cw[tr], M[vf], yc[vf], cw[vf])
    return models, M


def evaluate(models, X, M, test_mask, gold):
    R = {}
    fin = X["is_final"].to_numpy() == 1
    te = test_mask & fin
    y = X["kept"].to_numpy()
    pf_ = X["portfolio"].to_numpy()
    for name in ["pre_call", "in_call", "post_call"]:
        m = models[name]
        raw = m.raw(M[te])
        cal = m.predict(M[te], pf_[te])
        R[f"keep_{name}"] = dict(
            auc=round(roc_auc_score(y[te], cal), 4), auc_raw=round(roc_auc_score(y[te], raw), 4),
            ece_raw=round(ece(y[te], raw), 4), ece_calibrated=round(ece(y[te], cal), 4),
            brier=round(brier_score_loss(y[te], cal), 4), n=int(te.sum()), base_rate=round(y[te].mean(), 4))
        log(f"[keep:{name}] {R[f'keep_{name}']}")
    # per-portfolio calibration of in-call model
    cal = models["in_call"].predict(M[te], pf_[te])
    rows = []
    for p in sorted(set(pf_[te])):
        mm = pf_[te] == p
        rows.append(dict(portfolio=p, n=int(mm.sum()), keep_rate=round(y[te][mm].mean(), 3),
                         mean_score=round(cal[mm].mean(), 3), ece=round(ece(y[te][mm], cal[mm], 5), 3),
                         auc=round(roc_auc_score(y[te][mm], cal[mm]), 3) if 0 < y[te][mm].mean() < 1 else None))
    R["calibration_by_portfolio"] = rows
    log("[keep:in_call] per-portfolio calibration:", pd.DataFrame(rows).to_string(index=False))

    # category model (end of call)
    cm = models["category"]
    Pc = cm.proba(M[te])
    yt = X["y_cat"].to_numpy()[te]
    pred_argmax = Pc.argmax(1)
    pred_cost = cm.decide(Pc)
    R["category_vs_ground_truth"] = _cls_report(yt, pred_cost, Pc, "cost-sensitive decision (test, GT=gold+pseudo)")
    R["category_argmax_vs_ground_truth"] = _cls_report(yt, pred_argmax, Pc, "argmax (test, GT=gold+pseudo)")
    try:
        R["category_ovr_auc_macro"] = round(roc_auc_score(yt, Pc, multi_class="ovr", average="macro"), 4)
    except ValueError:
        R["category_ovr_auc_macro"] = None
    R["category_ece_top1"] = round(multiclass_ece(yt, Pc), 4)
    # vs human annotations that fall in the test split (fully independent of training)
    ids = X.loc[te, "ptp_id"].to_numpy()
    gm = pd.Series(ids).isin(gold["ptp_id"]).to_numpy()
    if gm.sum():
        gy = pd.Series(ids[gm]).map(gold.set_index("ptp_id")["annotated_type"]).map(CAT2ID).to_numpy()
        R["category_vs_human_test"] = dict(n=int(gm.sum()), acc=round(accuracy_score(gy, pred_cost[gm]), 3),
                                           macro_f1=round(f1_score(gy, pred_cost[gm], average="macro"), 3))
        log("[category] vs human annotations in test split:", R["category_vs_human_test"])
    # hardship misclassification (key PS metric) before and after guardrails
    gi = CAT2ID["genuine_infeasible"]
    agg = [CAT2ID["escape"], CAT2ID["repeat_promiser"]]
    hm = yt == gi
    R["hardship_misclassified_as_aggressive_argmax"] = round(np.isin(pred_argmax[hm], agg).mean(), 4)
    R["hardship_misclassified_as_aggressive_costsens"] = round(np.isin(pred_cost[hm], agg).mean(), 4)
    log(f"[category] hardship->aggressive  argmax={R['hardship_misclassified_as_aggressive_argmax']}"
        f"  cost-sensitive={R['hardship_misclassified_as_aggressive_costsens']}")
    return R, Pc, pred_cost


def _cls_report(y, pred, Pc, title):
    labels = list(range(len(CATEGORIES)))
    rep = classification_report(y, pred, labels=labels, target_names=CATEGORIES, digits=3, zero_division=0,
                                output_dict=True)
    txt = classification_report(y, pred, labels=labels, target_names=CATEGORIES, digits=3, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    log(f"[category] {title}\n{txt}\nconfusion (rows=true, cols=pred, order={CATEGORIES}):\n{cm}")
    return dict(report=rep, confusion=cm.tolist())


# =========================================================================== streaming evaluation
def streaming_eval(models, X, M, test_mask):
    """Quality vs how much of the call has been heard; 'heard enough' trigger."""
    te = test_mask & X["call_id"].notna().to_numpy()
    Xt, Mt = X[te].copy(), M[te]
    Xt["p_keep"] = models["in_call"].predict(Mt, Xt["portfolio"].to_numpy())
    Pc = models["category"].proba(Mt)
    Xt["cat_pred"] = models["category"].decide(Pc)
    Xt["cat_conf"] = Pc.max(1)
    tot = Xt.groupby("ptp_id")["n_turns"].transform("max")
    Xt["frac"] = Xt["n_turns"] / tot
    rows = []
    for lo, hi in [(0, .25), (.25, .5), (.5, .75), (.75, .99), (.99, 1.01)]:
        # last snapshot in each fraction bucket per call
        s = Xt[(Xt["frac"] > lo) & (Xt["frac"] <= hi)].groupby("ptp_id").tail(1)
        if len(s) < 30 or s["kept"].nunique() < 2:
            continue
        rows.append(dict(call_fraction=f"{int(lo * 100)}-{int(min(hi, 1) * 100)}%", n=len(s),
                         keep_auc=round(roc_auc_score(s["kept"], s["p_keep"]), 4),
                         keep_ece=round(ece(s["kept"], s["p_keep"]), 4),
                         category_acc=round(accuracy_score(s["y_cat"], s["cat_pred"]), 4),
                         category_macro_f1=round(f1_score(s["y_cat"], s["cat_pred"], average="macro"), 4)))
    curve = pd.DataFrame(rows)
    log("[streaming] quality vs share of call heard:\n" + curve.to_string(index=False))

    # 'heard enough' trigger: commitment / record event seen AND prediction stable over 2 customer turns
    Xt = Xt.sort_values(["ptp_id", "n_turns"])
    g = Xt.groupby("ptp_id")
    Xt["dp"] = g["p_keep"].diff().abs()
    Xt["same_cat"] = (g["cat_pred"].shift(1) == Xt["cat_pred"]).astype(int)
    Xt["stable"] = (Xt["dp"].fillna(1) < 0.05) & (Xt["same_cat"] == 1) & (Xt["cat_conf"] >= 0.5)
    Xt["event"] = (Xt["n_c_commit"] + Xt["n_a_record"] + Xt["n_a_propose_date"] + Xt["n_c_third_party"]) > 0
    Xt["trigger"] = Xt["stable"] & Xt["event"]
    first = Xt[Xt["trigger"]].groupby("ptp_id").head(1).set_index("ptp_id")
    final = g.tail(1).set_index("ptp_id")
    trig = final.join(first[["n_turns", "frac", "cat_pred", "p_keep", "elapsed_s"]], rsuffix="_trig")
    fired = trig["n_turns_trig"].notna()
    tr = trig[fired]
    agree = (tr["cat_pred_trig"] == tr["cat_pred"]).mean()
    trig_stats = dict(
        calls=int(len(trig)), trigger_fired_share=round(fired.mean(), 4),
        median_turn_fraction_at_trigger=round(tr["frac_trig"].median(), 3),
        median_seconds_before_call_end=round((tr["elapsed_s"] - tr["elapsed_s_trig"]).median(), 1),
        category_agreement_trigger_vs_final=round(agree, 4),
        keep_auc_at_trigger=round(roc_auc_score(tr["kept"], tr["p_keep_trig"]), 4),
        category_acc_at_trigger=round(accuracy_score(tr["y_cat"], tr["cat_pred_trig"]), 4),
    )
    log("[streaming] heard-enough trigger:", trig_stats)
    return curve, trig_stats


# =========================================================================== cross-fitted scores for every PTP
def crossfit_scores(X, conv_cols, folds=5):
    """Out-of-fold predictions for ALL PTPs (account-grouped folds), so the
    enriched CSV never shows in-sample scores."""
    acc_ids = X["account_id"].unique()
    rng = np.random.RandomState(SEED)
    fold_of = dict(zip(acc_ids, rng.randint(0, folds, len(acc_ids))))
    f = X["account_id"].map(fold_of).to_numpy()
    fin = X["is_final"].to_numpy() == 1
    out = []
    for k in range(folds):
        test = f == k
        rest_acc = [a for a in acc_ids if fold_of[a] != k]
        va_acc = set(rng.choice(rest_acc, int(0.18 * len(rest_acc)), replace=False))
        va = (~test) & X["account_id"].isin(va_acc).to_numpy()
        tr = (~test) & ~va
        models, M = train_stage_models(X, conv_cols, tr, va)
        sel = test & fin
        pf_ = X["portfolio"].to_numpy()[sel]
        Pc = models["category"].proba(M[sel])
        o = pd.DataFrame({"ptp_id": X.loc[sel, "ptp_id"].to_numpy(),
                          "p_keep_pre_call": models["pre_call"].predict(M[sel], pf_),
                          "p_keep_in_call": models["in_call"].predict(M[sel], pf_),
                          "p_keep_post_call": models["post_call"].predict(M[sel], pf_),
                          "cat_pred_id": models["category"].decide(Pc),
                          "cat_conf": Pc.max(1)})
        for i, c in enumerate(CATEGORIES):
            o["p_" + c] = Pc[:, i]
        sh = models["in_call"].shap(M[sel])
        o["shap_top"] = _shap_reasons(sh, M[sel])
        out.append(o)
        log(f"[crossfit] fold {k} done ({sel.sum()} ptps)")
    return pd.concat(out, ignore_index=True)


READABLE = {
    "total_broken_ratio": "share of past promises broken", "total_prior_broken": "number of past broken promises",
    "prev_broken_ratio": "past broken-promise ratio", "ability_to_pay_estimate": "ability-to-pay estimate",
    "amount_to_overdue": "promised amount vs overdue", "amount_x_atp": "promise size vs ability to pay",
    "is_salary_window": "promise date falls in salary window", "is_pre_salary_hazard": "promise date just before salary",
    "days_after_salary": "days between salary and promise date", "discussed_day_vs_salary": "discussed date vs salary day",
    "n_c_hedge": "borrower hedging ('dekhta hoon', 'try karunga')", "n_c_exit": "borrower trying to end call",
    "n_c_firm": "firm commitment language", "n_c_date": "borrower stated a date", "n_c_partial_offer": "borrower offered part payment",
    "recorded_without_cust_date": "PTP recorded but borrower never stated a date", "n_a_pressure": "agent pressure phrases",
    "agent_date_no_cust_date": "only the agent proposed the date", "n_c_hardship": "hardship mentioned",
    "n_c_conditional": "conditional promise ('salary aayi toh')", "n_c_vague_date": "vague date ('next week')",
    "agent_keep_rate": "agent's historical keep rate", "agent_keep_lift": "agent keep rate vs peers",
    "promise_lag_days": "days until promised date", "pay_cnt_30d": "payments in last 30 days",
    "bureau_ord": "bureau score band", "dpd_start": "days past due", "n_c_link_accept": "borrower asked for payment link",
    "n_c_link_refuse": "borrower refused payment link", "min_ack_ratio": "share of minimal 'haan/ok' replies",
    "cust_pause_mean": "borrower response latency", "n_long_hesitation": "long hesitations before replying",
    "cust_asr_mean": "speech clarity (ASR confidence)", "n_c_third_party": "someone other than the borrower answered",
    "n_c_prior_break_excuse": "excuse for a past broken promise", "n_a_prior_break": "agent cited a past broken promise",
    "hist_broken": "broken promises in the observed period", "answer_rate_30d": "answer rate in last 30 days",
}


def _shap_reasons(sh, M, k=3):
    out = []
    vals = sh.to_numpy()
    cols = sh.columns
    for i in range(len(sh)):
        idx = np.argsort(-np.abs(vals[i]))[:k]
        parts = []
        for j in idx:
            name = READABLE.get(cols[j], cols[j])
            v = M.iloc[i][cols[j]]
            vs = f"{v:.2f}" if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v)
            parts.append(f"{name}={vs} ({'+' if vals[i, j] > 0 else '-'}{abs(vals[i, j]):.2f})")
        out.append("; ".join(parts))
    return out


# =========================================================================== decisions + audits
def decide_all(scores, X, gt):
    fin = X[X["is_final"] == 1].set_index("ptp_id")
    rows = []
    for r in scores.itertuples(index=False):
        f = fin.loc[r.ptp_id]
        feats = f.to_dict()
        cat_model = CATEGORIES[r.cat_pred_id]
        cat, rails = P.apply_guardrails(cat_model, feats)
        pdate = pd.Timestamp(f["promised_date"]).strftime("%d %b")
        utt = P.guidance(cat, f["preferred_language"], feats, r.p_keep_in_call, f["promised_amount"], pdate,
                         f["salary_credit_day"])
        issues = P.compliance_check(utt, cat)
        park, rem, queue, fu = P.parking_plan(cat, r.p_keep_in_call, f["promise_lag_days"], rails)
        rows.append(dict(ptp_id=r.ptp_id, category_model=cat_model, predicted_category=cat,
                         guardrails="|".join(rails), next_utterance=utt, compliance_issues="|".join(issues),
                         parking_days=park, reminders="|".join(rem), queue=queue, follow_up=fu,
                         explanation=_explain(cat, feats, r)))
    return pd.DataFrame(rows)


def _explain(cat, f, r):
    g = lambda k: f.get(k, 0) if f.get(k) == f.get(k) else 0
    bits = []
    if g("n_c_third_party"):
        bits.append("non-borrower answered the call")
    if g("recorded_without_cust_date"):
        bits.append("borrower never stated a date; agent proposed it")
    if g("n_a_pressure"):
        bits.append(f"{int(g('n_a_pressure'))} pressure phrase(s) before the promise")
    if g("n_c_hedge") or g("n_c_exit"):
        bits.append(f"hedging/exit phrases x{int(g('n_c_hedge') + g('n_c_exit'))}")
    if g("n_c_severe_hardship"):
        bits.append("severe hardship mentioned (medical/job loss)")
    elif g("n_c_hardship"):
        bits.append("hardship mentioned")
    if g("n_c_conditional"):
        bits.append("conditional promise")
    if g("total_prior_broken") >= 2:
        bits.append(f"{int(g('total_prior_broken'))} past broken promises")
    if g("is_pre_salary_hazard"):
        bits.append("date falls 1-4 days before salary")
    if g("is_salary_window"):
        bits.append("date within 5 days after salary")
    if g("n_c_firm") or g("n_c_partial_offer"):
        bits.append("firm commitment / part-payment offer by borrower")
    s = f"{cat} (conf {r.cat_conf:.2f}); keep p={r.p_keep_in_call:.2f}. Signals: " + ("; ".join(bits) or "none")
    return s + ". Score drivers: " + r.shap_top


def historical_disclosure_audit(ts):
    """Baseline DPDP risk in the historical calls: debt details spoken by the agent/bot
    AFTER a third party identified themselves."""
    t = ts.copy()
    t["tp_seen"] = t.groupby("call_id")["c_third_party"].cumsum() > 0
    after = t[t["tp_seen"] & (t["is_c"] == 0)]
    disc = after[after["text"].str.contains(r"₹|\brs\b|rupaye|rupees|ptp|overdue|emi|promise", case=False, regex=True)]
    calls_tp = t[t["c_third_party"] > 0]["call_id"].nunique()
    return dict(calls_with_third_party=int(calls_tp), calls_with_post_identification_disclosure=int(disc["call_id"].nunique()),
                share=round(disc["call_id"].nunique() / max(calls_tp, 1), 4))


def agent_dashboard(enriched, base):
    x = enriched.merge(base[["ptp_id", "agent_shift_end_share", "agent_generic_remark_share"]], on="ptp_id")
    x["agent_keep_lift"] = x.groupby("agent_id")["kept"].transform("mean") - x.groupby("channel")["kept"].transform("mean")
    d = x.groupby(["agent_id", "channel"]).agg(
        raw_ptps=("ptp_id", "size"), quality_weighted_ptps=("p_keep_in_call", "sum"), actual_kept=("kept", "sum"),
        pushed_share=("predicted_category", lambda s: (s == "agent_recorded_or_pushed").mean()),
        shift_end_share=("agent_shift_end_share", "first"), generic_remark_share=("agent_generic_remark_share", "first"),
        keep_lift_vs_peers=("agent_keep_lift", "first")).reset_index()
    d["quality_ratio"] = d["quality_weighted_ptps"] / d["raw_ptps"]
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    tele = d["channel"] == "tele_agent"
    d["integrity_risk_z"] = np.nan
    d.loc[tele, "integrity_risk_z"] = (z(d.loc[tele, "pushed_share"]) + z(d.loc[tele, "generic_remark_share"]) +
                                       z(d.loc[tele, "shift_end_share"]) - z(d.loc[tele, "quality_ratio"])) / 4
    d["integrity_flag"] = d["integrity_risk_z"] > 0.75
    return d.sort_values("integrity_risk_z", ascending=False)


def psi(a, b, bins=10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return 0.0
    ea = np.histogram(a, q)[0] / len(a) + 1e-4
    eb = np.histogram(b, q)[0] / len(b) + 1e-4
    return float(((ea - eb) * np.log(ea / eb)).sum())


# =========================================================================== business impact
def business_impact(en, scen=(0.08, 0.15, 0.22)):
    """Projection on held-out PTPs. Observed: outcomes, promise lags. Assumed: conversion
    of flagged weak promises by in-call firming / early follow-up (scenario grid; to be
    measured in the RCT). Costs from config.COST."""
    t = en[en["split"] == "test"].copy()
    n = len(t)
    broken = t["kept"] == 0
    weak_q = {"early_followup", "senior_caller", "integrity_review", "rpc_confirmation"}
    flagged = t["queue"].isin(weak_q) | (t["p_keep_in_call"] < 0.35)
    base_park = t["promise_lag_days"].clip(lower=0)
    dyn_park = t["parking_days"]
    rec = dict(
        n_test_ptps=n, observed_keep_rate=round(1 - broken.mean(), 4),
        broken_flagged_before_parking=round(flagged[broken].mean(), 4),
        kept_flagged_false_alarm=round(flagged[~broken].mean(), 4),
        precision_of_flag=round((~t.loc[flagged, "kept"].astype(bool)).mean(), 4),
        contact_days_unlocked_per_1000_ptps=round(((base_park - dyn_park).clip(lower=0)[broken]).sum() / n * 1000, 1),
        avg_parking_days_baseline=round(base_park.mean(), 2), avg_parking_days_dynamic=round(dyn_park.mean(), 2),
    )
    # escalation cost: incumbent sends broken PTPs to tele re-attempts (3 connects) and 35% to field
    esc_cost = 3 * COST["tele_caller"] + 0.35 * COST["field_visit"]
    fb = (flagged & broken)
    false_alarm_cost = (flagged & ~broken).sum() * (COST["voice_bot"] + COST["sms_whatsapp"])
    scen_rows = []
    for s in scen:
        converted = fb.sum() * s
        avg_amt = t.loc[fb, "promised_amount"].mean()
        token_amt = t.loc[fb, "promised_amount"].map(P.token_amount).mean()
        scen_rows.append(dict(
            conversion_of_flagged_broken=s,
            projected_keep_rate=round((t["kept"].sum() + converted) / n, 4),
            broken_rate_reduction_pp=round(converted / n * 100, 2),
            escalation_cost_avoided_per_1000=round(converted * esc_cost / n * 1000, 0),
            false_alarm_cost_per_1000=round(false_alarm_cost / n * 1000, 0),
            recovery_lift_per_1000_rs_low=round(converted * token_amt / n * 1000, 0),
            recovery_lift_per_1000_rs_high=round(converted * avg_amt / n * 1000, 0),
        ))
    rec["scenarios"] = scen_rows
    # RCT sizing: detect +3pp keep-rate lift, alpha .05 two-sided, power .8
    p0 = 1 - broken.mean()
    p1 = p0 + 0.03
    pbar = (p0 + p1) / 2
    from scipy.stats import norm
    za, zb = norm.ppf(0.975), norm.ppf(0.8)
    nn = ((za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) / (p1 - p0)) ** 2
    rec["rct_ptps_per_arm_for_3pp"] = int(np.ceil(nn))
    log("[business]", json.dumps(rec, indent=1, default=float))
    return rec


# =========================================================================== main
def main(fast=False):
    T0 = time.time()
    D, lab, ts, pf, base, conv_cols = build()
    gt = ground_truth(D, base, pf)
    gold_res, _ = gold_cv(D, base, pf) if not fast else ({}, None)
    X = snapshot_frame(base, pf, ts, gt)
    split = X["split"].to_numpy()
    tr, va, te = split == "train", split == "validation", split == "test"
    log(f"[data] snapshots={len(X)}  ptps: train={X[tr].ptp_id.nunique()} val={X[va].ptp_id.nunique()} "
        f"test={X[te].ptp_id.nunique()}  keep-rate={lab.kept.mean():.3f}")
    models, M = train_stage_models(X, conv_cols, tr, va)
    R, _, _ = evaluate(models, X, M, te, D["annotations"])
    curve, trig = streaming_eval(models, X, M, te)
    R["streaming_curve"] = curve.to_dict("records")
    R["heard_enough_trigger"] = trig
    R["label_pipeline_gold_cv"] = gold_res
    R["label_distribution"] = gt["category"].value_counts().to_dict()
    R["label_sources"] = gt["category_source"].value_counts().to_dict()
    R["outcome_status"] = lab["status"].value_counts().to_dict()
    R["kept_rate"] = round(lab["kept"].mean(), 4)
    R["cured_rate"] = round(lab["cured"].mean(), 4)

    # global feature importance (mean |SHAP|) of the in-call model on test
    fin_te = te & (X["is_final"].to_numpy() == 1)
    sh = models["in_call"].shap(M[fin_te])
    imp = sh.abs().mean().sort_values(ascending=False)
    R["in_call_top_features"] = imp.head(25).round(4).to_dict()
    log("[shap] top in-call drivers:\n" + imp.head(15).round(4).to_string())
    sh_post = models["post_call"].shap(M[fin_te]).abs().mean().sort_values(ascending=False)
    R["post_call_top_features"] = sh_post.head(15).round(4).to_dict()

    with open(MODEL_DIR / "models.pkl", "wb") as fh:
        pickle.dump(dict(models=models, conv_cols=conv_cols, cat_levels=F._CAT_LEVELS), fh)

    # drift monitor (train vs test) on top drivers
    drift = {c: round(psi(M.loc[tr & (X.is_final == 1).to_numpy(), c].astype(float),
                          M.loc[fin_te, c].astype(float)), 4)
             for c in imp.head(15).index if pd.api.types.is_numeric_dtype(M[c])}
    R["drift_psi_train_vs_test"] = drift

    # cross-fitted scores + decisions for every PTP
    scores = crossfit_scores(X, conv_cols)
    dec = decide_all(scores, X, gt)
    en = base[["ptp_id", "account_id", "attempt_id", "agent_id", "channel", "split", "portfolio",
               "preferred_language", "promised_date", "promised_amount", "promise_lag_days", "kept", "cured",
               "status", "pay_ratio_grace"]] \
        .merge(gt[["ptp_id", "category", "category_source", "category_conf", "rule_id"]]
               .rename(columns={"category": "ground_truth_category", "category_conf": "ground_truth_conf"}),
               on="ptp_id") \
        .merge(scores, on="ptp_id").merge(dec, on="ptp_id")
    en["confidence_score"] = en["p_keep_in_call"].round(3)
    en["category_confidence"] = en["cat_conf"].round(3)
    R["crossfit_all_ptps"] = dict(
        keep_auc_in_call=round(roc_auc_score(en.kept, en.p_keep_in_call), 4),
        keep_auc_pre_call=round(roc_auc_score(en.kept, en.p_keep_pre_call), 4),
        keep_auc_post_call=round(roc_auc_score(en.kept, en.p_keep_post_call), 4),
        keep_ece_in_call=round(ece(en.kept, en.p_keep_in_call), 4),
        category_acc=round(accuracy_score(en.ground_truth_category, en.category_model), 4),
        category_macro_f1=round(f1_score(en.ground_truth_category, en.category_model, average="macro"), 4))
    log("[crossfit] all-PTP out-of-fold:", R["crossfit_all_ptps"])

    # audits
    hs = (en["guardrails"].str.contains("hardship|vulnerable"))
    fin_all = X[X["is_final"] == 1].set_index("ptp_id")
    hardship_any = en["ptp_id"].map((fin_all["n_c_hardship"].fillna(0) + fin_all["hardship_30d"].fillna(0)) > 0)
    aggressive_q = en["queue"].isin(["senior_caller", "early_followup"]) & en["predicted_category"].isin(["escape", "repeat_promiser"])
    tp_any = en["ptp_id"].map(fin_all["n_c_third_party"].fillna(0) > 0)
    R["audit"] = dict(
        hardship_signal_ptps=int(hardship_any.sum()),
        hardship_routed_to_aggressive_after_guardrails=int((hardship_any & aggressive_q).sum()),
        hardship_softened_by_guardrail=int(hs.sum()),
        third_party_ptps=int(tp_any.sum()),
        third_party_utterances_with_debt_disclosure=int(en.loc[tp_any, "compliance_issues"]
                                                       .str.contains("DPDP").sum()),
        any_utterance_with_banned_language=int(en["compliance_issues"].str.contains("FPC_abusive").sum()),
        historical_agent_disclosure_after_3p_identified=historical_disclosure_audit(ts),
    )
    log("[audit]", json.dumps(R["audit"], indent=1))
    R["business"] = business_impact(en)
    dash = agent_dashboard(en, base)
    dash.to_csv(OUT_DIR / "agent_integrity_dashboard.csv", index=False)

    cols = ["ptp_id", "account_id", "agent_id", "channel", "split", "portfolio", "promised_date", "promised_amount",
            "predicted_category", "category_model", "category_confidence", "confidence_score", "p_keep_pre_call",
            "p_keep_in_call", "p_keep_post_call"] + ["p_" + c for c in CATEGORIES] + \
           ["explanation", "guardrails", "next_utterance", "compliance_issues", "parking_days", "reminders", "queue",
            "follow_up", "ground_truth_category", "category_source", "ground_truth_conf", "rule_id", "kept", "cured",
            "status", "pay_ratio_grace"]
    en[cols].to_csv(OUT_DIR / "enriched_ptp_predictions.csv", index=False, encoding="utf-8-sig")
    gt.to_csv(OUT_DIR / "ground_truth_labels.csv", index=False)
    lab.to_csv(OUT_DIR / "outcome_labels.csv", index=False)
    R["runtime_s"] = round(time.time() - T0, 1)
    with open(OUT_DIR / "results.json", "w") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (OUT_DIR / "pipeline_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"[done] {R['runtime_s']}s -> {OUT_DIR}")
    return R


if __name__ == "__main__":
    import sys
    main(fast="--fast" in sys.argv)
