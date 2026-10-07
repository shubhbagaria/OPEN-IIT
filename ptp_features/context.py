"""
Account and agent context features.

Two very different risk profiles live in this file.

ACCOUNT FEATURES are mostly joins and arithmetic. The only subtlety is the
circular month arithmetic for salary dates, and a leakage check on the
`accounts.csv` columns that lack a `_start` suffix.

AGENT FEATURES are where target leakage usually enters a project like this.
"Agent TA005's historical PTP-to-payment rate" is a powerful feature and a
trap, because the naive version uses outcomes from the future. Two guards are
applied here:

  1. TIME. A statistic attached to a call at time t uses only promises whose
     outcome was already KNOWABLE at t -- that is, promised_date + grace < t.
     A promise due next Tuesday tells you nothing today, even though it sits
     earlier in the file.

  2. SPLIT. By default only train-split accounts contribute history at all, so
     no information crosses from validation or test into a training row.

Guard 1 is the one most pipelines get wrong. A plain `groupby().mean()` over
the whole file fails both.
"""

from __future__ import annotations

import math
from calendar import monthrange

import numpy as np
import pandas as pd

from .config import Config

# --------------------------------------------------------------------------
# Account features
# --------------------------------------------------------------------------

# Columns in accounts.csv that are explicitly point-in-time snapshots.
POINT_IN_TIME_COLS = ("bucket_start", "dpd_start", "overdue_start")

# Columns with no `_start` suffix whose as-of semantics are NOT documented.
# audit.py tests these empirically and reports the verdict; until that comes
# back clean they are passed through but flagged in the feature dictionary.
UNVERIFIED_AS_OF_COLS = (
    "prev_ptp_count",
    "prev_ptp_broken",
    "ability_to_pay_estimate",
    "paid_other_lenders_30d",
    "other_active_loans",
    "last_bounce_reason",
    "bureau_score_band",
)


def _days_before_salary(promised_date: pd.Timestamp, salary_day: float) -> float:
    """
    How many days BEFORE the borrower's salary credit the promise falls,
    measured circularly within the month.

      0  -> promise lands exactly on salary day
      1..5 -> promise is just before payday (structurally hard to keep)
      25..30 -> promise is just after payday (easy)

    Circular arithmetic matters: a promise on the 29th against a salary day of
    the 2nd is three days before payday, not twenty-seven days after it.
    """
    if pd.isna(promised_date) or salary_day is None or pd.isna(salary_day):
        return math.nan
    days_in_month = monthrange(promised_date.year, promised_date.month)[1]
    salary_day = int(min(max(int(salary_day), 1), days_in_month))
    return float((salary_day - promised_date.day) % days_in_month)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Numeric column, or an all-NaN series when the optional column is absent."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _flag(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """
    0/1 flag as nullable Int64 that stays <NA> when any input is missing.

    A bare `(x < 0.99).astype(...)` turns a missing x into 0, which tells the
    model "known to be false" when the truth is "unknown".
    """
    out = condition.astype("Int64")
    missing = pd.Series(False, index=condition.index)
    for s in inputs:
        missing |= s.isna()
    return out.mask(missing)


def build_account_features(
    ptps: pd.DataFrame, accounts: pd.DataFrame, lenders: pd.DataFrame
) -> pd.DataFrame:
    """
    One row per ptp_id carrying account context and promise-vs-capacity ratios.

    All of these are known at the moment the promise is captured, so every
    column here is safe for the in-call model.
    """
    df = ptps.merge(accounts, on="account_id", how="left", validate="many_to_one")
    df = df.merge(
        lenders[["lender_id", "lender_type"]],
        on="lender_id",
        how="left",
        validate="many_to_one",
    )

    captured = pd.to_datetime(df["captured_ts"], errors="coerce")
    promised = pd.to_datetime(df["promised_date"], errors="coerce")

    # --- promise shape --------------------------------------------------
    df["acct_promise_horizon_days"] = (promised - captured.dt.normalize()).dt.days
    df["acct_promised_amount"] = pd.to_numeric(df["promised_amount"], errors="coerce")
    df["acct_overdue_at_capture"] = pd.to_numeric(
        df["overdue_at_capture"], errors="coerce"
    )

    # Below 1.0 means the borrower promised only part of what is owed. That is
    # not a bad sign in itself -- a realistic part payment often keeps better
    # than an unrealistic full one.
    df["acct_promise_to_overdue_ratio"] = (
        df["acct_promised_amount"] / df["acct_overdue_at_capture"].replace(0, np.nan)
    )
    df["acct_is_part_payment_promise"] = _flag(
        df["acct_promise_to_overdue_ratio"] < 0.99, df["acct_promise_to_overdue_ratio"]
    )

    # --- capacity -------------------------------------------------------
    atp = _num(df, "ability_to_pay_estimate")
    emi = _num(df, "emi_amount")
    df["acct_promise_to_ability_ratio"] = df["acct_promised_amount"] / (
        atp.replace(0, np.nan) * emi.replace(0, np.nan)
    )
    df["acct_promise_to_emi_ratio"] = df["acct_promised_amount"] / emi.replace(0, np.nan)

    # --- salary timing --------------------------------------------------
    salary_day = _num(df, "salary_credit_day")
    df["acct_has_salary_day"] = salary_day.notna().astype(int)
    df["acct_days_promised_before_salary"] = [
        _days_before_salary(p, s) for p, s in zip(promised, salary_day)
    ]
    # The structural-infeasibility flag: promised money just before payday.
    days_before = df["acct_days_promised_before_salary"]
    df["acct_promise_before_payday"] = _flag(
        days_before.between(1, 5, inclusive="both"), days_before
    )

    # --- promise history ------------------------------------------------
    prev_n = _num(df, "prev_ptp_count")
    prev_broken = _num(df, "prev_ptp_broken")
    df["acct_prev_ptp_count"] = prev_n
    df["acct_prev_ptp_broken"] = prev_broken
    df["acct_prev_ptp_broken_rate"] = prev_broken / prev_n.replace(0, np.nan)
    df["acct_is_repeat_promiser"] = _flag(
        (prev_n >= 2) & (prev_broken >= 2), prev_n, prev_broken
    )

    # prev_ptp_count is a window-start snapshot (constant per account; see the
    # leakage audit), so it misses promises made inside the window. This is the
    # running count from ptps.csv, strictly before this promise was captured.
    order = pd.DataFrame({"account_id": df["account_id"], "captured": captured})
    order = order.sort_values("captured", kind="stable")
    prior = order.groupby("account_id").cumcount()
    # Ties on captured_ts would otherwise count each other; take the first rank.
    tie_first = order.groupby(["account_id", "captured"]).cumcount()
    df["acct_n_prior_ptps_in_data"] = (prior - tie_first).reindex(df.index)
    df.loc[captured.isna(), "acct_n_prior_ptps_in_data"] = np.nan

    # --- who answered ---------------------------------------------------
    answerer = df.get("answerer_recorded")
    if answerer is not None:
        df["acct_answerer_is_self"] = (
            answerer.astype(str).str.lower().eq("self").astype("Int64")
        )
        df["acct_answerer"] = answerer.astype(str).str.lower()

    link = df.get("payment_link_sent")
    if link is not None:
        df["acct_payment_link_sent"] = (
            link.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        )

    # --- timing of the call itself --------------------------------------
    df["acct_capture_hour"] = captured.dt.hour
    df["acct_capture_dow"] = captured.dt.dayofweek
    df["acct_capture_is_late_shift"] = _flag(captured.dt.hour >= 18, captured)

    passthrough = [
        c
        for c in (
            "portfolio", "income_type", "town_id", "preferred_language",
            "bucket_start", "dpd_start", "emi_amount", "outstanding",
            "bureau_score_band", "other_active_loans", "paid_other_lenders_30d",
            "last_bounce_reason", "ability_to_pay_estimate", "lender_id",
            "lender_type", "dialling_arm", "channel", "source",
        )
        if c in df.columns
    ]
    engineered = [c for c in df.columns if c.startswith("acct_")]

    return df[["ptp_id", "account_id", "agent_id"] + passthrough + engineered].copy()


# --------------------------------------------------------------------------
# Agent features -- the leakage-sensitive half
# --------------------------------------------------------------------------


def build_agent_features(
    ptps: pd.DataFrame,
    attempts: pd.DataFrame,
    agents: pd.DataFrame,
    accounts: pd.DataFrame,
    splits: pd.DataFrame,
    labels: pd.DataFrame | None,
    cfg: Config,
) -> pd.DataFrame:
    """
    Per-PTP agent context, every statistic computed strictly as-of the call.

    `labels` is optional: without it the outcome-based features are emitted as
    NaN and everything else still works. Expected columns: ptp_id, kept.
    """
    fc = cfg.features

    agent_static = agents.rename(
        columns={
            "channel": "agent_channel",
            "language_team": "agent_language_team",
            "town_id": "agent_town_id",
            "tenure_months": "agent_tenure_months",
            "shift": "agent_shift",
        }
    )

    df = ptps[["ptp_id", "account_id", "agent_id", "captured_ts", "promised_date"]].copy()
    df["captured_ts"] = pd.to_datetime(df["captured_ts"], errors="coerce")
    df["promised_date"] = pd.to_datetime(df["promised_date"], errors="coerce")
    df = df.merge(agent_static, on="agent_id", how="left", validate="many_to_one")

    # Language match between the agent's team and the borrower's preference.
    df = df.merge(
        accounts[["account_id", "preferred_language"]],
        on="account_id",
        how="left",
        validate="many_to_one",
    )
    team = df["agent_language_team"].astype("string").str.lower()
    pref = df["preferred_language"].astype("string").str.lower()
    df["agent_language_match"] = _flag(team == pref, team, pref)
    df.loc[team == "all", "agent_language_match"] = 1
    df = df.drop(columns=["preferred_language"])

    df["agent_is_bot"] = (
        df["agent_channel"].astype(str).str.lower().eq("voice_bot").astype("Int64")
    )

    # ---- which accounts may contribute history --------------------------
    split_map = splits.set_index("account_id")["split"]
    if fc.agent_history_pool == "train":
        allowed_accounts = set(split_map[split_map == "train"].index)
    elif fc.agent_history_pool == "all":
        allowed_accounts = set(split_map.index)
    else:
        raise ValueError(f"Unknown agent_history_pool: {fc.agent_history_pool!r}")

    # ---- history table ---------------------------------------------------
    hist = ptps[["ptp_id", "account_id", "agent_id", "captured_ts", "promised_date"]].copy()
    hist["captured_ts"] = pd.to_datetime(hist["captured_ts"], errors="coerce")
    hist["promised_date"] = pd.to_datetime(hist["promised_date"], errors="coerce")
    hist = hist[hist["account_id"].isin(allowed_accounts)]

    if labels is not None and not labels.empty:
        hist = hist.merge(labels[["ptp_id", "kept"]], on="ptp_id", how="left")
    else:
        hist["kept"] = np.nan

    # The moment this promise's outcome became knowable.
    if fc.respect_outcome_known_time:
        hist["outcome_known_ts"] = cfg.label.window_end(hist["promised_date"])
    else:
        hist["outcome_known_ts"] = hist["captured_ts"]

    # When the call itself started. The attempt row's timestamp precedes
    # captured_ts by a few minutes, so "attempts before captured_ts" would
    # count the call's own attempt (and its own PTP) as history. Field visits
    # have no attempt and fall back to captured_ts.
    call_start = ptps[["ptp_id", "attempt_id"]].merge(
        attempts[["attempt_id", "attempt_ts"]], on="attempt_id", how="left"
    ).drop_duplicates("ptp_id")
    df = df.merge(call_start[["ptp_id", "attempt_ts"]], on="ptp_id", how="left")
    df["_call_start_ts"] = pd.to_datetime(df["attempt_ts"], errors="coerce").fillna(
        df["captured_ts"]
    )
    df = df.drop(columns=["attempt_ts"])

    df = _attach_asof_agent_rates(df, hist, min_history=fc.min_agent_history)
    df = _attach_asof_volume(df, attempts, allowed_accounts, fc.min_agent_history)
    df = _attach_bunching_and_remarks(df, attempts, allowed_accounts)

    # agent_id itself is an identifier, already carried by the account table;
    # keeping it here too produced agent_id_x / agent_id_y after the join.
    keep = ["ptp_id"] + [
        c for c in df.columns if c.startswith("agent_") and c != "agent_id"
    ]
    return df[keep].copy()


def _attach_asof_agent_rates(
    df: pd.DataFrame, hist: pd.DataFrame, *, min_history: int
) -> pd.DataFrame:
    """
    Agent PTP-to-payment rate, using only promises already resolved at call time.

    Implemented as a merge_asof on `outcome_known_ts`, which is the clean way to
    express "everything this agent had been proven right or wrong about by now".
    """
    resolved = hist.dropna(subset=["outcome_known_ts", "kept"]).sort_values(
        "outcome_known_ts"
    )

    if resolved.empty:
        df["agent_hist_kept_rate"] = np.nan
        df["agent_hist_n_resolved"] = 0
        return df

    # Running totals per agent, ordered by when each outcome became known.
    resolved = resolved.copy()
    resolved["cum_kept"] = resolved.groupby("agent_id")["kept"].cumsum()
    resolved["cum_n"] = resolved.groupby("agent_id").cumcount() + 1

    left = df.sort_values("captured_ts").reset_index()
    merged = pd.merge_asof(
        left,
        resolved[["outcome_known_ts", "agent_id", "cum_kept", "cum_n"]],
        left_on="captured_ts",
        right_on="outcome_known_ts",
        by="agent_id",
        direction="backward",
        allow_exact_matches=False,  # strictly before the call
    )

    rate = merged["cum_kept"] / merged["cum_n"]
    rate[merged["cum_n"].fillna(0) < min_history] = np.nan  # too thin to trust

    merged["agent_hist_kept_rate"] = rate
    merged["agent_hist_n_resolved"] = merged["cum_n"].fillna(0).astype(int)

    return (
        merged.set_index("index")
        .sort_index()
        .drop(columns=["outcome_known_ts", "cum_kept", "cum_n"])
    )


def _attach_asof_volume(
    df: pd.DataFrame,
    attempts: pd.DataFrame,
    allowed_accounts: set,
    min_history: int,
) -> pd.DataFrame:
    """
    How often this agent converts a connected call into a logged promise --
    computed from attempts that started strictly before the current call
    started (so the call's own attempt never counts).

    No outcome is involved, so only the time guard applies. An unusually high
    rate is the first signal of agent-pushed promises.
    """
    att = attempts[attempts["account_id"].isin(allowed_accounts)].copy()
    att["attempt_ts"] = pd.to_datetime(att["attempt_ts"], errors="coerce")
    att = att.dropna(subset=["attempt_ts"]).sort_values("attempt_ts")

    connected = att["talk_duration_s"].fillna(0) > 0
    att["is_connect"] = connected.astype(int)
    att["is_ptp"] = att["ptp_id"].notna().astype(int)

    att["cum_connects"] = att.groupby("agent_id")["is_connect"].cumsum()
    att["cum_ptps"] = att.groupby("agent_id")["is_ptp"].cumsum()

    left = df.sort_values("_call_start_ts").reset_index()
    merged = pd.merge_asof(
        left,
        att[["attempt_ts", "agent_id", "cum_connects", "cum_ptps"]],
        left_on="_call_start_ts",
        right_on="attempt_ts",
        by="agent_id",
        direction="backward",
        allow_exact_matches=False,
    )

    rate = merged["cum_ptps"] / merged["cum_connects"].replace(0, np.nan)
    rate[merged["cum_connects"].fillna(0) < min_history] = np.nan

    merged["agent_hist_ptp_per_connect"] = rate
    merged["agent_hist_n_connects"] = merged["cum_connects"].fillna(0).astype(int)

    return (
        merged.set_index("index")
        .sort_index()
        .drop(columns=["attempt_ts", "cum_connects", "cum_ptps"])
    )


def _attach_bunching_and_remarks(
    df: pd.DataFrame, attempts: pd.DataFrame, allowed_accounts: set
) -> pd.DataFrame:
    """
    Two integrity signals from the brief.

    BUNCHING: promises clustered into the last working hour, which is what
    logging promises to hit a daily target looks like in the data.

    REMARK REUSE: the share of an agent's remarks that are copies of other
    remarks they have written. Copy-paste remarks accompany promises that were
    never really taken.

    Both are deliberately computed per agent-DAY and then lagged, so a row only
    ever sees the agent's completed previous days, never its own day.
    """
    att = attempts[attempts["account_id"].isin(allowed_accounts)].copy()
    att["attempt_ts"] = pd.to_datetime(att["attempt_ts"], errors="coerce")
    att = att.dropna(subset=["attempt_ts"])
    att["day"] = att["attempt_ts"].dt.normalize()
    att["hour"] = att["attempt_ts"].dt.hour
    att["is_ptp"] = att["ptp_id"].notna().astype(int)

    daily = (
        att.groupby(["agent_id", "day"])
        .agg(
            n_ptps=("is_ptp", "sum"),
            n_attempts=("attempt_id", "count"),
            n_ptps_late=("is_ptp", lambda s: int(s[att.loc[s.index, "hour"] >= 18].sum())),
            n_remarks=("remark", lambda s: int(s.notna().sum())),
            n_distinct_remarks=("remark", lambda s: int(s.dropna().nunique())),
        )
        .reset_index()
    )

    daily["late_ptp_share"] = daily["n_ptps_late"] / daily["n_ptps"].replace(0, np.nan)
    daily["remark_reuse_rate"] = 1.0 - (
        daily["n_distinct_remarks"] / daily["n_remarks"].replace(0, np.nan)
    )

    # Expanding mean over the agent's days up to and INCLUDING each day. The
    # call then takes the latest such row from a day strictly before its own
    # (merge_asof, no exact matches), which is "all previous days" even when
    # the agent had no history-pool activity on the call day itself. An exact
    # same-day join would leave those rows empty.
    daily = daily.sort_values(["agent_id", "day"])
    for src, dst in (
        ("late_ptp_share", "agent_late_shift_bunching"),
        ("remark_reuse_rate", "agent_remark_reuse_rate"),
    ):
        daily[dst] = daily.groupby("agent_id")[src].transform(
            lambda s: s.expanding().mean()
        )

    df = df.copy()
    df["_day"] = pd.to_datetime(df["captured_ts"], errors="coerce").dt.normalize()
    left = df.sort_values("_day").reset_index()
    out = pd.merge_asof(
        left,
        daily[["day", "agent_id", "agent_late_shift_bunching", "agent_remark_reuse_rate"]]
        .sort_values("day"),
        left_on="_day",
        right_on="day",
        by="agent_id",
        direction="backward",
        allow_exact_matches=False,  # previous days only, never the call's own day
    )
    return out.set_index("index").sort_index().drop(columns=["_day", "day"])
