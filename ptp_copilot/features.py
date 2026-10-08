"""Multi-modal feature engineering.

Four blocks, each strictly time-aware (nothing observed after the decision
point is allowed into the block that is used at that decision point):

  A. account / pre-call         - known before the dial (accounts, history before capture)
  B. conversation + acoustic    - computed on a transcript *prefix* (live, turn-by-turn)
  C. agent integrity            - agent-level behaviour, estimated out-of-fold on train only
  D. post-call (<= 36h)         - payment-link / WhatsApp engagement, for the parking decision
"""
import numpy as np
import pandas as pd

from . import lexicon as lx
from .config import GRACE_DAYS, CURE_DAYS

BUREAU_ORD = {"300-549": 0, "550-649": 1, "650-699": 2, "700-749": 3, "750-900": 4}
BUCKET_ORD = {"X": 0, "1-30": 1, "31-60": 2, "61-90": 3, "90+": 4}


# =========================================================== B. per-turn signals
def turn_signals(tr: pd.DataFrame) -> pd.DataFrame:
    """One row per turn with binary lexical signals. Vectorised over the corpus."""
    t = tr.reset_index(drop=True)
    txt = t["text"].map(lx.clean)
    is_c = (t["speaker"] == "customer").to_numpy()
    n = {}  # new columns, assembled once at the end
    for name, rx in lx.CUSTOMER_FAMILIES.items():
        n[f"c_{name}"] = (txt.str.contains(rx).to_numpy() & is_c).astype(np.int8)
    for name, rx in lx.AGENT_FAMILIES.items():
        n[f"a_{name}"] = (txt.str.contains(rx).to_numpy() & ~is_c).astype(np.int8)
    n["c_min_ack"] = (txt.map(lx.is_min_ack).to_numpy(bool) & is_c).astype(np.int8)
    sal = txt.map(lx.extract_salary_day).to_numpy(float)
    n["salary_day_said"] = np.where(is_c, sal, np.nan)
    days = txt.map(lx.extract_days)
    rel = txt.str.contains(lx.DATE_RELATIVE).to_numpy()
    # a "date" turn is an explicit day or relative day, excluding salary-day statements
    has_date = ((days.str.len() > 0).to_numpy() | rel) & np.isnan(sal)
    prior = n["a_prior_break"] > 0
    n["c_date"] = (has_date & is_c).astype(np.int8)
    # dates inside "you promised 12 Jan last time" are history, not a proposal
    n["a_date"] = (has_date & ~is_c & ~prior).astype(np.int8)
    day = np.array([d[-1] if len(d) else np.nan for d in days], dtype=float)
    n["day_said"] = np.where(~np.isnan(sal) | prior, np.nan, day)
    amts = txt.map(lx.extract_amounts).str.len().to_numpy() > 0
    n["c_amount"] = (amts & is_c).astype(np.int8)
    n["a_amount"] = (amts & ~is_c).astype(np.int8)
    n["n_words"] = txt.str.split().str.len().fillna(0).to_numpy()
    n["inaudible"] = t["text"].str.count(r"\[inaudible\]").to_numpy()
    n["dur_ms"] = (t["end_ms"] - t["start_ms"]).clip(lower=0).to_numpy()
    n["is_c"] = is_c.astype(np.int8)
    # commitment: customer turn with a date / firm / min-ack following an agent date proposal
    cid = t["call_id"]
    shift = lambda a, k: pd.Series(a).groupby(cid).shift(k).fillna(0).to_numpy()
    prev_prop = shift(n["a_propose_date"], 1)
    prev_pressure = shift(n["a_pressure"], 1) + shift(n["a_pressure"], 2)
    n["c_ack_to_proposal"] = ((prev_prop > 0) & (n["c_min_ack"] > 0)).astype(np.int8)
    n["c_reply_to_proposal"] = ((prev_prop > 0) & is_c).astype(np.int8)
    n["pause_reply_to_proposal"] = np.where(n["c_reply_to_proposal"] > 0, t["pause_before_ms"], np.nan)
    n["c_commit"] = (is_c & ((n["c_date"] > 0) | (n["c_firm"] > 0) | (n["c_ack_to_proposal"] > 0))).astype(np.int8)
    n["c_commit_after_pressure"] = ((n["c_commit"] > 0) & (prev_pressure > 0)).astype(np.int8)
    return pd.concat([t, pd.DataFrame(n)], axis=1)


CUM_COLS = ([f"c_{k}" for k in lx.CUSTOMER_FAMILIES] + [f"a_{k}" for k in lx.AGENT_FAMILIES] +
            ["c_min_ack", "c_date", "a_date", "c_amount", "a_amount", "c_ack_to_proposal",
             "c_reply_to_proposal", "c_commit", "c_commit_after_pressure", "inaudible"])


def prefix_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Cumulative conversation features for EVERY prefix (row k = turns 0..k of the call).

    The same function serves offline training (all prefixes) and live inference
    (the last row of the prefix seen so far)."""
    g = ts.groupby("call_id", sort=False)
    f = {}  # dict of Series -> one DataFrame at the end (much faster than column inserts)
    f["call_id"] = ts["call_id"]
    f["turn_idx"] = ts["turn_idx"]
    f["n_turns"] = g.cumcount() + 1
    for c in CUM_COLS:
        f["n_" + c] = g[c].cumsum()
    f["n_cust_turns"] = g["is_c"].cumsum()
    f["n_agent_turns"] = f["n_turns"] - f["n_cust_turns"]
    cw = ts["n_words"] * ts["is_c"]
    aw = ts["n_words"] * (1 - ts["is_c"])
    f["cust_words"] = cw.groupby(ts["call_id"]).cumsum()
    f["agent_words"] = aw.groupby(ts["call_id"]).cumsum()
    f["talk_listen_ratio"] = f["cust_words"] / (f["agent_words"] + 1)
    cd = ts["dur_ms"] * ts["is_c"]
    f["cust_talk_share"] = cd.groupby(ts["call_id"]).cumsum() / (g["dur_ms"].cumsum() + 1)
    f["elapsed_s"] = ts["end_ms"] / 1000.0
    f["min_ack_ratio"] = f["n_c_min_ack"] / f["n_cust_turns"].clip(lower=1)

    # acoustic: customer pauses / ASR confidence (expanding stats over customer turns only)
    cp = ts["pause_before_ms"].where(ts["is_c"] > 0)
    ca = ts["asr_confidence"].where(ts["is_c"] > 0)
    f["cust_pause_mean"] = cp.groupby(ts["call_id"]).transform(lambda s: s.expanding().mean())
    ffill = lambda s: s.groupby(ts["call_id"]).ffill()
    f["cust_pause_max"] = ffill(cp.groupby(ts["call_id"]).cummax())
    f["n_long_hesitation"] = (cp > 1200).astype(int).groupby(ts["call_id"]).cumsum()
    f["cust_asr_mean"] = ca.groupby(ts["call_id"]).transform(lambda s: s.expanding().mean())
    f["cust_pause_mean"] = ffill(f["cust_pause_mean"])
    f["cust_asr_mean"] = ffill(f["cust_asr_mean"])
    f["cust_asr_min"] = ffill(ca.groupby(ts["call_id"]).cummin())
    f["pause_reply_to_proposal_max"] = ffill(ts["pause_reply_to_proposal"].groupby(ts["call_id"]).cummax())
    f["last_cust_pause"] = cp.groupby(ts["call_id"]).ffill()

    # recency: signals in the last 3 turns (promise right before hanging up / exit)
    for c in ["c_exit", "c_hedge", "c_min_ack", "c_commit", "a_pressure", "a_record"]:
        f["last3_" + c] = g[c].transform(lambda s: s.rolling(3, min_periods=1).sum())

    # date attribution: who first mentioned a payment date
    first_c = ffill(ts["turn_idx"].where(ts["c_date"] > 0).groupby(ts["call_id"]).cummin())
    first_a = ffill(ts["turn_idx"].where((ts["a_date"] > 0) | (ts["a_propose_date"] > 0)).groupby(ts["call_id"]).cummin())
    f["date_first_by_customer"] = ((first_c.notna()) & ((first_a.isna()) | (first_c < first_a))).astype(int)
    f["date_first_by_agent"] = ((first_a.notna()) & ((first_c.isna()) | (first_a < first_c))).astype(int)
    f["agent_date_no_cust_date"] = ((f["n_a_date"] + f["n_a_propose_date"] > 0) & (f["n_c_date"] == 0)).astype(int)
    f["recorded_without_cust_date"] = ((f["n_a_record"] > 0) & (f["n_c_date"] == 0)).astype(int)
    f["recorded_without_cust_amount"] = ((f["n_a_record"] > 0) & (f["n_c_amount"] == 0) & (f["n_c_partial_offer"] == 0)).astype(int)
    f["recorded_without_commit"] = ((f["n_a_record"] > 0) & (f["n_c_commit"] == 0)).astype(int)
    # day under discussion = latest explicit day mentioned by anyone; did the customer say it?
    dd = ts["day_said"].groupby(ts["call_id"]).ffill()
    f["day_under_discussion"] = dd
    cust_days = ts["day_said"].where(ts["is_c"] > 0)
    f["cust_last_day"] = cust_days.groupby(ts["call_id"]).ffill()
    f["cust_said_discussed_day"] = (f["cust_last_day"] == f["day_under_discussion"]).astype(int)
    f["salary_day_said"] = ts["salary_day_said"].groupby(ts["call_id"]).ffill()
    f["first_commit_turn"] = ffill(ts["turn_idx"].where(ts["c_commit"] > 0).groupby(ts["call_id"]).cummin())
    f["turns_since_first_commit"] = f["turn_idx"] - f["first_commit_turn"]
    f["negotiation_rounds"] = f["n_a_partial_ask"] + f["n_c_partial_offer"]
    f["hedge_exit_score"] = f["n_c_hedge"] + f["n_c_exit"] + f["n_c_forgot"] + 0.5 * f["n_c_vague_date"]
    f["firmness_score"] = f["n_c_firm"] + f["n_c_date"] + f["n_c_partial_offer"] + f["n_c_link_accept"]
    f["speaker_is_bot"] = g["speaker"].transform(lambda s: (s == "bot").any()).astype(int)
    return pd.DataFrame(f, index=ts.index)


def call_level(pf: pd.DataFrame) -> pd.DataFrame:
    """Final row of every call = end-of-call feature vector."""
    return pf.groupby("call_id", sort=False).tail(1).set_index("call_id")


def snapshot_rows(pf: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """Training snapshots for the streaming model: every customer turn + the final turn."""
    last = pf.groupby("call_id")["n_turns"].transform("max") == pf["n_turns"]
    keep = (ts["is_c"].to_numpy() > 0) | last.to_numpy()
    out = pf[keep].copy()
    out["is_final"] = last[keep].astype(int)
    return out


# =========================================================== A. account / pre-call
def _days_from_salary(day, sal_day):
    """Signed days from salary credit to promised day-of-month, in (-15, 15]."""
    d = (day - sal_day) % 30
    return np.where(np.isnan(sal_day), np.nan, np.where(d > 15, d - 30, d))


def account_features(D, labels: pd.DataFrame) -> pd.DataFrame:
    p = D["ptps"].copy()
    a = D["accounts"].copy()
    a["bureau_ord"] = a["bureau_score_band"].map(BUREAU_ORD)
    a["bucket_ord"] = a["bucket_start"].map(BUCKET_ORD)
    a["prev_broken_ratio"] = a["prev_ptp_broken"] / a["prev_ptp_count"].replace(0, np.nan)
    a["overdue_to_outstanding"] = a["overdue_start"] / a["outstanding"].replace(0, np.nan)
    a["overdue_to_emi"] = a["overdue_start"] / a["emi_amount"].replace(0, np.nan)
    a["paid_other_lenders_30d"] = a["paid_other_lenders_30d"].astype(int)
    keep = ["account_id", "lender_id", "portfolio", "income_type", "town_id", "preferred_language", "bucket_ord",
            "dpd_start", "emi_amount", "overdue_start", "outstanding", "salary_credit_day", "bureau_ord",
            "other_active_loans", "paid_other_lenders_30d", "last_bounce_reason", "ability_to_pay_estimate",
            "prev_ptp_count", "prev_ptp_broken", "prev_broken_ratio", "overdue_to_outstanding", "overdue_to_emi"]
    p = p.merge(a[keep], on="account_id", how="left")

    p["promise_lag_days"] = (p["promised_date"] - p["captured_ts"].dt.normalize()).dt.days
    p["amount_to_overdue"] = p["promised_amount"] / p["overdue_at_capture"].replace(0, np.nan)
    p["amount_to_emi"] = p["promised_amount"] / p["emi_amount"].replace(0, np.nan)
    p["amount_x_atp"] = p["amount_to_overdue"] * (1 - p["ability_to_pay_estimate"].fillna(0.5))
    p["promised_dow"] = p["promised_date"].dt.dayofweek
    p["promised_dom"] = p["promised_date"].dt.day
    p["capture_hour"] = p["captured_ts"].dt.hour + p["captured_ts"].dt.minute / 60
    p["capture_dow"] = p["captured_ts"].dt.dayofweek
    p["answerer_not_self"] = (p["answerer_recorded"] != "self").astype(int)
    p["payment_link_sent"] = p["payment_link_sent"].astype(int)
    d = _days_from_salary(p["promised_dom"].to_numpy(float), p["salary_credit_day"].to_numpy(float))
    p["days_after_salary"] = d
    p["is_salary_window"] = ((d >= 0) & (d <= 5)).astype(int)
    p["is_pre_salary_hazard"] = ((d >= -4) & (d <= -1)).astype(int)
    p["salary_known"] = p["salary_credit_day"].notna().astype(int)

    # ---- in-data account history strictly before capture (no leakage):
    # only earlier PTPs whose cure window has fully closed count as "observed".
    lab = labels[["ptp_id", "kept", "cure_end", "settle_days"]]
    h = p[["ptp_id", "account_id", "captured_ts"]].merge(lab, on="ptp_id")
    hist = h.merge(h[["account_id", "captured_ts", "ptp_id"]], on="account_id", suffixes=("_old", ""))
    hist = hist[hist["ptp_id_old"] != hist["ptp_id"]]
    seen = hist[hist["captured_ts_old"] < hist["captured_ts"]]
    closed = seen[seen["cure_end"] < seen["captured_ts"]]
    agg_seen = seen.groupby("ptp_id").agg(hist_ptps_any=("ptp_id_old", "size"),
                                          last_prev_ptp_ts=("captured_ts_old", "max"))
    agg_closed = closed.groupby("ptp_id").agg(hist_ptps_closed=("kept", "size"), hist_kept=("kept", "sum"),
                                              hist_settle_days=("settle_days", "mean"))
    p = p.merge(agg_seen, on="ptp_id", how="left").merge(agg_closed, on="ptp_id", how="left")
    for c in ["hist_ptps_any", "hist_ptps_closed", "hist_kept"]:
        p[c] = p[c].fillna(0)
    p["hist_broken"] = p["hist_ptps_closed"] - p["hist_kept"]
    p["total_prior_ptps"] = p["prev_ptp_count"] + p["hist_ptps_any"]
    p["total_prior_broken"] = p["prev_ptp_broken"] + p["hist_broken"]
    p["total_broken_ratio"] = p["total_prior_broken"] / p["total_prior_ptps"].replace(0, np.nan)
    p["days_since_prev_ptp"] = (p["captured_ts"] - p["last_prev_ptp_ts"]).dt.days
    p = p.drop(columns=["last_prev_ptp_ts"])

    # ---- dialler history before capture
    dl = D["dials"][["account_id", "attempt_ts", "network_response", "disposition"]]
    x = p[["ptp_id", "account_id", "captured_ts"]].merge(dl, on="account_id")
    x = x[(x["attempt_ts"] < x["captured_ts"]) & (x["attempt_ts"] >= x["captured_ts"] - pd.Timedelta(days=30))]
    dh = x.groupby("ptp_id").agg(
        dial_30d=("attempt_ts", "size"),
        answered_30d=("network_response", lambda s: (s == "answered").sum()),
        refused_30d=("disposition", lambda s: s.isin(["rpc_refused", "call_rejected", "rpc_hung_up"]).sum()),
        tp_contact_30d=("disposition", lambda s: s.isin(["third_party_contact", "third_party_ptp"]).sum()),
        hardship_30d=("disposition", lambda s: (s == "rpc_hardship").sum()),
    )
    p = p.merge(dh, on="ptp_id", how="left")
    for c in dh.columns:
        p[c] = p[c].fillna(0)
    p["answer_rate_30d"] = p["answered_30d"] / p["dial_30d"].replace(0, np.nan)

    # ---- payments in the 30 days before capture (recent liquidity)
    py = D["payments"][["account_id", "payment_ts", "amount"]]
    y = p[["ptp_id", "account_id", "captured_ts"]].merge(py, on="account_id")
    y = y[(y["payment_ts"] < y["captured_ts"]) & (y["payment_ts"] >= y["captured_ts"] - pd.Timedelta(days=30))]
    ph = y.groupby("ptp_id").agg(pay_cnt_30d=("amount", "size"), pay_amt_30d=("amount", "sum"))
    p = p.merge(ph, on="ptp_id", how="left")
    p[["pay_cnt_30d", "pay_amt_30d"]] = p[["pay_cnt_30d", "pay_amt_30d"]].fillna(0)

    # ---- disposition remark at capture (agent free text)
    rem = D["dials"][["attempt_id", "remark", "disposition"]].rename(columns={"disposition": "capture_disposition"})
    p = p.merge(rem, on="attempt_id", how="left")
    return p


# =========================================================== C. agent integrity
GENERIC_REMARK = r"(?i)^(?:ptp liya|customer ne ptp diya|cust agreed|customer cooperative|ptp noted|follow up)"


def agent_integrity(p: pd.DataFrame, labels: pd.DataFrame, fit_mask=None, prior_k: float = 30.0):
    """Agent-level gaming / integrity indicators.

    Outcome-based indicators are strictly *time-aware*: for a PTP captured at t,
    the agent's keep rate uses only that agent's earlier PTPs whose grace window
    had already closed before t (exactly what a live system would know). The
    peer prior is the same quantity for the agent's channel cohort. This avoids
    the label leakage of leave-one-out target encoding."""
    x = p[["ptp_id", "agent_id", "channel", "captured_ts", "remark", "account_id"]].merge(
        labels[["ptp_id", "kept", "grace_end"]], on="ptp_id")

    def resolved_before(keys, t_now, t_res, kept):
        """For each row: (#resolved, #kept) among rows with same key and t_res < t_now."""
        n_out = np.zeros(len(t_now))
        k_out = np.zeros(len(t_now))
        for key in pd.unique(keys):
            m = np.where(keys == key)[0]
            order = np.argsort(t_res[m])
            tr_sorted = t_res[m][order]
            ck = np.concatenate([[0], np.cumsum(kept[m][order])])
            idx = np.searchsorted(tr_sorted, t_now[m], side="left")
            n_out[m] = idx
            k_out[m] = ck[idx]
        return n_out, k_out

    t_now = x["captured_ts"].to_numpy("datetime64[ns]")
    t_res = x["grace_end"].to_numpy("datetime64[ns]")
    kept = x["kept"].to_numpy(float)
    n_a, k_a = resolved_before(x["agent_id"].to_numpy(), t_now, t_res, kept)
    n_c, k_c = resolved_before(x["channel"].to_numpy(), t_now, t_res, kept)
    prior = np.where(n_c > 0, (k_c + 5 * 0.3) / (n_c + 5), 0.3)
    x["agent_resolved_ptps"] = n_a
    x["agent_keep_rate"] = (k_a + prior_k * prior) / (n_a + prior_k)
    x["agent_keep_lift"] = x["agent_keep_rate"] - prior

    # behavioural (outcome-free) indicators can use all rows
    x["capture_date"] = x["captured_ts"].dt.date
    last_ts = x.groupby(["agent_id", "capture_date"])["captured_ts"].transform("max")
    x["mins_before_agent_day_end"] = (last_ts - x["captured_ts"]).dt.total_seconds() / 60
    x["shift_end_ptp"] = ((x["captured_ts"].dt.hour >= 17) | (x["mins_before_agent_day_end"] <= 30)).astype(int)
    x["agent_shift_end_share"] = x.groupby("agent_id")["shift_end_ptp"].transform("mean")
    x["remark_generic"] = x["remark"].fillna("").str.contains(GENERIC_REMARK).astype(int)
    x["agent_generic_remark_share"] = x.groupby("agent_id")["remark_generic"].transform("mean")
    cnt = x.groupby("remark")["account_id"].transform("nunique")
    x["remark_copy_count"] = cnt.fillna(1)
    x["agent_ptps_per_day"] = x.groupby("agent_id")["ptp_id"].transform("size") / \
        x.groupby("agent_id")["capture_date"].transform("nunique")
    bot = x["agent_id"].str.startswith("BOT")
    x.loc[bot, ["agent_shift_end_share", "agent_generic_remark_share"]] = 0
    cols = ["agent_resolved_ptps", "agent_keep_rate", "agent_keep_lift", "mins_before_agent_day_end", "shift_end_ptp",
            "agent_shift_end_share", "remark_generic", "agent_generic_remark_share", "remark_copy_count",
            "agent_ptps_per_day"]
    return x[["ptp_id"] + cols]


# =========================================================== D. post-call engagement
def postcall_features(D, horizon_h: float = 36.0) -> pd.DataFrame:
    """Engagement within `horizon_h` hours after capture *and* before the promised date."""
    p = D["ptps"][["ptp_id", "captured_ts", "promised_date"]]
    ev = D["events"].merge(p, on="ptp_id")
    h = (ev["event_ts"] - ev["captured_ts"]).dt.total_seconds() / 3600
    ok = (h >= 0) & (h <= horizon_h) & (ev["event_ts"] < ev["promised_date"])
    ev = ev[ok].assign(h=h[ok])
    w = ev.pivot_table(index="ptp_id", columns="event_type", values="h", aggfunc="min")
    out = pd.DataFrame(index=p["ptp_id"])
    for e in ["payment_link_opened", "confirmation_reply_yes", "confirmation_reply_no", "inbound_call_from_customer",
              "whatsapp_confirmation_sent"]:
        col = w[e] if e in w else pd.Series(dtype=float)
        out["pc_" + e] = out.index.map(col).notna().astype(int)
    out["pc_hours_to_link_open"] = out.index.map(w["payment_link_opened"]) if "payment_link_opened" in w else np.nan
    return out.reset_index()


# =========================================================== model-frame helpers
CAT_FEATURES = ["portfolio", "income_type", "preferred_language", "last_bounce_reason", "channel",
                "answerer_recorded", "source", "capture_disposition", "lender_id"]
_CAT_LEVELS = {}


def to_model_frame(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Select model columns; string columns -> pandas category with frozen levels
    (so that training and live inference share the same encoding)."""
    out = df[cols].copy()
    for c in cols:
        is_cat = c in CAT_FEATURES or c in _CAT_LEVELS or (
            pd.api.types.is_string_dtype(out[c]) and not pd.api.types.is_numeric_dtype(out[c]))
        if is_cat:
            if c not in _CAT_LEVELS:
                _CAT_LEVELS[c] = sorted(out[c].dropna().astype(str).unique())
            out[c] = pd.Categorical(out[c].astype("string"), categories=_CAT_LEVELS[c])
        else:
            out[c] = pd.to_numeric(out[c].astype(float) if pd.api.types.is_bool_dtype(out[c]) else out[c],
                                   errors="coerce").astype(float)
    return out
