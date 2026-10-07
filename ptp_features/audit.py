"""
Audit: coverage, leakage, the feature dictionary, and annotation agreement.

This is the part of Phase 2 that teams skip and then regret. Four outputs:

  coverage_report.md    What fraction of promises have transcripts, what the
                        real language mix is, where the lexicon has holes.

  leakage_audit.md      Every feature with its time-availability class, plus an
                        empirical test of whether accounts.csv columns are
                        as-of-capture or end-of-period.

  feature_dictionary.csv  Name, group, definition, availability, null rate,
                          as-of caveat. Generated from the code so it cannot drift.

  annotation_check.md   How the features line up with the human taxonomy in
                        annotated_ptp_sample.csv (train + validation only).

Team C needs the dictionary more than they need extra features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import lexicons as lx
from .config import Config, DEFAULT
from .context import UNVERIFIED_AS_OF_COLS
from .features import CATEGORICAL_FEATURES

# --------------------------------------------------------------------------
# Time-availability classes
# --------------------------------------------------------------------------

AVAILABILITY = {
    "pre_call": "Known before the call starts. Safe everywhere.",
    "in_call": "Derived from turns 0..k. Safe at decision point k and later "
               "(in features_endcall, k is the last turn).",
    "post_call": "Arrives after the call. NEVER safe for the in-call model.",
    "meta": "Identifier or bookkeeping column, not a model input.",
}

META_COLUMNS = {
    "ptp_id", "account_id", "call_id", "agent_id", "split",
    "decision_point", "is_final_turn", "has_transcript", "k_turn_idx",
}


def classify_feature(name: str) -> str:
    if name in META_COLUMNS:
        return "meta"
    if name.startswith("post_"):
        return "post_call"
    if name.startswith("acct_") or name.startswith("agent_"):
        return "pre_call"
    if name in (
        "portfolio", "income_type", "town_id", "preferred_language",
        "bucket_start", "dpd_start", "emi_amount", "outstanding",
        "bureau_score_band", "other_active_loans", "paid_other_lenders_30d",
        "last_bounce_reason", "ability_to_pay_estimate", "lender_id",
        "lender_type", "dialling_arm", "channel", "source",
    ):
        return "pre_call"
    return "in_call"


# Columns built on accounts.csv fields whose as-of semantics are undocumented.
# The leakage audit tests prev_ptp_count empirically; the others cannot be
# tested from this data, so the dictionary carries the caveat to Team C.
UNVERIFIED_DERIVED = {
    "acct_prev_ptp_count": "prev_ptp_count",
    "acct_prev_ptp_broken": "prev_ptp_broken",
    "acct_prev_ptp_broken_rate": "prev_ptp_count, prev_ptp_broken",
    "acct_is_repeat_promiser": "prev_ptp_count, prev_ptp_broken",
    "acct_promise_to_ability_ratio": "ability_to_pay_estimate",
}


def as_of_caveat(name: str) -> str:
    if name in UNVERIFIED_AS_OF_COLS:
        return f"unverified as-of: {name}"
    if name in UNVERIFIED_DERIVED:
        return f"unverified as-of: {UNVERIFIED_DERIVED[name]}"
    return ""


FEATURE_GROUPS = {
    "lex_": "Lexical cue counts from borrower speech",
    "obs_lang_": "Observed spoken-language markers",
    "cust_": "Borrower speech volume and timing",
    "house_": "Bot / tele-agent behaviour",
    "n_": "Mention counts",
    "acct_": "Account context and promise-vs-capacity ratios",
    "agent_": "Agent context (as-of, train-pool only)",
    "post_": "Post-call engagement (quarantined)",
    "promise_": "Promise-turn characteristics",
    "first_": "Who proposed what, first",
    "pause_": "Promise-turn pause timing",
    "asr_": "ASR confidence around the promise",
    "latency_": "Response latency",
    "speech_": "Speech rate",
    "silence_": "Silence share",
    "turns_": "Turn-position counters",
    "safety_": "Hardship / refusal safety signals",
}


def feature_group(name: str) -> str:
    for prefix, label in FEATURE_GROUPS.items():
        if name.startswith(prefix):
            return label
    return "Other"


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def coverage(frames: dict[str, pd.DataFrame], outputs: dict[str, pd.DataFrame]) -> dict:
    """How much of the data the pipeline actually reached."""
    ptps = frames["ptps"]
    endcall = outputs["features_endcall"]
    incall = outputs["features_incall"]
    transcripts = frames["call_transcripts"]

    n_ptps = len(ptps)
    with_tx = int(endcall["has_transcript"].sum())

    report: dict = {
        "n_ptps": n_ptps,
        "n_ptps_with_transcript": with_tx,
        "pct_ptps_with_transcript": round(with_tx / n_ptps, 4) if n_ptps else 0.0,
        "n_ptps_from_calls": int(ptps["attempt_id"].notna().sum()),
        "n_ptps_from_visits": int(ptps["visit_id"].notna().sum())
        if "visit_id" in ptps.columns
        else 0,
        "n_distinct_calls_in_transcripts": int(transcripts["call_id"].nunique()),
        "n_incall_decision_points": len(incall),
        "mean_decision_points_per_ptp": (
            round(len(incall) / with_tx, 2) if with_tx else 0.0
        ),
    }

    # Turn counts
    turns_per_call = transcripts.groupby("call_id").size()
    report["turns_per_call"] = {
        "min": int(turns_per_call.min()) if len(turns_per_call) else 0,
        "median": float(turns_per_call.median()) if len(turns_per_call) else 0.0,
        "max": int(turns_per_call.max()) if len(turns_per_call) else 0,
    }

    # Language coverage -- the headline Phase 2 objective.
    cust = transcripts[transcripts["speaker"].astype(str).str.lower() == "customer"]
    norm = [lx.normalise(t) for t in cust["text"].fillna("")]
    report["lexicon_coverage_overall"] = lx.coverage_report(norm)

    # Coverage broken down by the account's registered language, to expose a
    # lexicon that works for one language and not another.
    if "account_id" in frames["accounts"].columns:
        att = frames["dial_attempts"][["attempt_id", "account_id"]].rename(
            columns={"attempt_id": "call_id"}
        )
        lang = frames["accounts"][["account_id", "preferred_language"]]
        tagged = cust.merge(att, on="call_id", how="left").merge(
            lang, on="account_id", how="left"
        )
        per_lang = {}
        for language, grp in tagged.groupby("preferred_language"):
            per_lang[str(language)] = lx.coverage_report(
                [lx.normalise(t) for t in grp["text"].fillna("")]
            )
        report["lexicon_coverage_by_registered_language"] = per_lang

    # Null rates on the end-call table.
    nulls = endcall.isna().mean().sort_values(ascending=False)
    # Promises sourced from a call whose call_id has no transcript rows at all.
    with_attempt = ptps["attempt_id"].notna()
    report["n_call_ptps_missing_transcript"] = int(
        (with_attempt & ~ptps["attempt_id"].isin(transcripts["call_id"])).sum()
    )

    report["highest_null_features"] = {
        k: round(float(v), 4) for k, v in nulls.head(25).items()
    }

    # Split integrity.
    splits = frames["splits"]
    report["split_sizes"] = splits["split"].value_counts().to_dict()
    report["accounts_in_multiple_splits"] = int(
        splits.groupby("account_id")["split"].nunique().gt(1).sum()
    )

    return report


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------


def test_accounts_as_of(frames: dict[str, pd.DataFrame]) -> dict:
    """
    Empirically test whether accounts.csv history columns are as-of-capture.

    Method: for every account, count the promises that actually precede each
    promise in the file, and compare that to `prev_ptp_count`. If the column is
    an as-of snapshot taken at the start of the observation window, it should
    be roughly CONSTANT per account and should NOT track the running count. If
    it tracks the running count, it is per-promise as-of. If it equals the
    account's TOTAL promise count, it was computed over the whole window and
    it LEAKS.

    This is the check the work plan flagged; here it runs automatically.
    """
    ptps = frames["ptps"].copy()
    ptps["captured_ts"] = pd.to_datetime(ptps["captured_ts"], errors="coerce")
    ptps = ptps.sort_values(["account_id", "captured_ts"])
    ptps["running_prior"] = ptps.groupby("account_id").cumcount()

    totals = ptps.groupby("account_id").size().rename("total_ptps")
    acc = frames["accounts"][["account_id"]].copy()
    for col in ("prev_ptp_count", "prev_ptp_broken"):
        if col in frames["accounts"].columns:
            acc[col] = frames["accounts"][col]

    merged = ptps.merge(acc, on="account_id", how="left").merge(
        totals, on="account_id", how="left"
    )

    out: dict = {}
    if "prev_ptp_count" in merged.columns:
        valid = merged.dropna(subset=["prev_ptp_count"])
        if len(valid):
            out["corr_with_running_prior"] = round(
                float(valid["prev_ptp_count"].corr(valid["running_prior"])), 4
            )
            out["corr_with_total_in_window"] = round(
                float(valid["prev_ptp_count"].corr(valid["total_ptps"])), 4
            )
            out["pct_equal_to_total_in_window"] = round(
                float((valid["prev_ptp_count"] == valid["total_ptps"]).mean()), 4
            )
            out["constant_within_account"] = bool(
                valid.groupby("account_id")["prev_ptp_count"].nunique().le(1).all()
            )

            # Verdict
            if out["pct_equal_to_total_in_window"] > 0.8:
                out["verdict"] = (
                    "LIKELY LEAKS -- prev_ptp_count matches the account's total "
                    "promise count across the whole window. Do not use as-is; "
                    "recompute a running prior from ptps.csv instead."
                )
            elif out["corr_with_running_prior"] > 0.8:
                out["verdict"] = "SAFE -- behaves like a per-promise running prior."
            elif out["constant_within_account"]:
                out["verdict"] = (
                    "SAFE (snapshot) -- constant per account, consistent with a "
                    "window-start snapshot. Prefer a recomputed running prior "
                    "for promises after the first."
                )
            else:
                out["verdict"] = "UNCLEAR -- inspect manually before using."
    out["columns_with_undocumented_as_of_semantics"] = list(UNVERIFIED_AS_OF_COLS)
    return out


def leak_canary(
    table: pd.DataFrame, labels: pd.DataFrame, cfg: Config
) -> list[tuple[str, float]]:
    """
    Flag any in-call feature correlating suspiciously hard with the label.

    A smoke alarm, not a rule. A genuinely strong feature can trip it; the
    point is that somebody looks.
    """
    merged = table.merge(labels[["ptp_id", "kept"]], on="ptp_id", how="inner")
    # Only look at train-split labels: an audit that reads test outcomes is
    # itself a way of tuning on the test set.
    if "split" in merged.columns:
        merged = merged[merged["split"] == "train"]
    if merged.empty:
        return []

    flagged: list[tuple[str, float]] = []
    for col in merged.columns:
        if col in META_COLUMNS or col == "kept":
            continue
        if classify_feature(col) != "in_call":
            continue
        series = pd.to_numeric(merged[col], errors="coerce")
        if series.notna().sum() < 50 or series.nunique(dropna=True) < 2:
            continue
        corr = series.corr(merged["kept"])
        if corr is not None and not math.isnan(corr):
            if abs(corr) >= cfg.features.leak_canary_abs_corr:
                flagged.append((col, round(float(corr), 4)))

    return sorted(flagged, key=lambda r: abs(r[1]), reverse=True)


def _label_source(labels: pd.DataFrame) -> str:
    if "label_source" in labels.columns and len(labels):
        return ", ".join(sorted(labels["label_source"].astype(str).unique()))
    return "unknown"


# --------------------------------------------------------------------------
# Annotation agreement
# --------------------------------------------------------------------------

# For each taxonomy class, the feature-level rule that should fire on it. These
# are deliberately simple: the point is to check that the features carry the
# signal the taxonomy describes, not to build a classifier here. Each rule is an
# OR over (column, value) pairs.
DETECTORS: dict[str, tuple[tuple[tuple[str, int], ...], str]] = {
    "agent_recorded_or_pushed": (
        (("cust_stated_neither_date_nor_amount", 1),),
        "borrower never stated a date or an amount",
    ),
    "third_party": (
        (("lex_third_party_any", 1), ("acct_answerer_is_self", 0)),
        "third-party cue, or answerer not recorded as self",
    ),
    "genuine_infeasible": (
        (("acct_promise_before_payday", 1), ("safety_signal_any", 1),
         ("lex_conditional_any", 1)),
        "promise just before payday, hardship/refusal cue, or conditional cue",
    ),
    "repeat_promiser": (
        (("acct_is_repeat_promiser", 1),),
        "2+ earlier promises and 2+ broken (accounts snapshot)",
    ),
    "escape": (
        (("lex_termination_any", 1), ("lex_link_refusal_any", 1),
         ("cust_stated_vague_date_only", 1)),
        "termination cue, link refusal, or only a vague date",
    ),
}

# Features whose per-class means are shown side by side.
PROFILE_FEATURES = (
    "has_transcript",
    "cust_stated_neither_date_nor_amount",
    "cust_stated_concrete_date",
    "cust_stated_vague_date_only",
    "lex_hedge_any",
    "lex_conditional_any",
    "lex_hardship_any",
    "lex_third_party_any",
    "lex_termination_any",
    "lex_link_refusal_any",
    "safety_signal_any",
    "acct_answerer_is_self",
    "acct_promise_before_payday",
    "acct_is_repeat_promiser",
    "acct_prev_ptp_broken_rate",
)


def annotation_check(
    frames: dict[str, pd.DataFrame], outputs: dict[str, pd.DataFrame]
) -> dict:
    """
    Compare the engineered features with the human taxonomy annotations.

    Uses train + validation rows only. The test split is excluded so that
    anyone tuning lexicons or rules from this report is not tuning on test.
    Returns {} when annotated_ptp_sample.csv is absent.
    """
    ann = frames.get("annotated_ptp_sample")
    if ann is None or ann.empty:
        return {}

    endcall = outputs["features_endcall"]
    labels = outputs.get("labels", pd.DataFrame(columns=["ptp_id", "kept"]))

    df = ann.merge(endcall, on="ptp_id", how="left").merge(
        labels[["ptp_id", "kept"]], on="ptp_id", how="left"
    )
    out: dict = {
        "n_annotated": int(len(ann)),
        "n_unmatched_ptp_ids": int(df["split"].isna().sum()),
        "n_by_split": df["split"].fillna("unmatched").value_counts().to_dict(),
        "label_source": _label_source(labels),
    }
    df = df[df["split"].isin(["train", "validation"])]
    out["n_used"] = int(len(df))
    if df.empty:
        return out

    out["n_by_confidence"] = df["annotator_confidence"].value_counts().to_dict()

    # Kept rate per class: genuine_feasible should sit well above the rest.
    kept = df.groupby("annotated_type")["kept"].agg(["mean", "count"])
    out["kept_rate_by_type"] = {
        t: {"kept_rate": round(float(r["mean"]), 4), "n": int(r["count"])}
        for t, r in kept.iterrows()
    }

    profile = {}
    for feat in PROFILE_FEATURES:
        if feat not in df.columns:
            continue
        vals = pd.to_numeric(df[feat], errors="coerce")
        profile[feat] = {
            t: (None if pd.isna(v) else round(float(v), 3))
            for t, v in vals.groupby(df["annotated_type"]).mean().items()
        }
    out["feature_means_by_type"] = profile

    det = {}
    is_type = df["annotated_type"]
    for cls, (terms, desc) in DETECTORS.items():
        if not all(col in df.columns for col, _ in terms):
            continue
        fired = pd.Series(False, index=df.index)
        for col, val in terms:
            fired |= pd.to_numeric(df[col], errors="coerce").eq(val).fillna(False)
        is_cls = is_type == cls
        tp = int((fired & is_cls).sum())
        det[cls] = {
            "rule": " OR ".join(f"{c} == {v}" for c, v in terms),
            "meaning": desc,
            "n_class": int(is_cls.sum()),
            "n_fired": int(fired.sum()),
            "recall": round(tp / int(is_cls.sum()), 3) if is_cls.any() else None,
            "precision": round(tp / int(fired.sum()), 3) if fired.any() else None,
        }
    out["detectors"] = det
    return out


def _annotation_markdown(ann: dict) -> str:
    lines = [
        "# Annotation check", "",
        "How the engineered features line up with the human taxonomy in "
        "`annotated_ptp_sample.csv`. The annotation is a label: it is used here "
        "for validation only and never enters a feature table.", "",
        f"- Annotated promises: **{ann['n_annotated']}** "
        f"(by split: {ann['n_by_split']})",
        f"- Annotated ptp_ids not found in ptps.csv: {ann['n_unmatched_ptp_ids']}",
        f"- Used here (train + validation; **test excluded** so nobody tunes on "
        f"it): **{ann.get('n_used', 0)}**",
        f"- Outcome label source: `{ann['label_source']}`",
    ]
    if not ann.get("n_used"):
        return "\n".join(lines)

    lines += ["", "## Classes", "", "| Type | n | Kept rate |", "| --- | --- | --- |"]
    for t, r in sorted(ann["kept_rate_by_type"].items()):
        lines.append(f"| {t} | {r['n']} | {r['kept_rate']:.1%} |")
    lines += ["", f"Annotator confidence: {ann['n_by_confidence']}"]

    lines += [
        "", "## Rule-level detectors", "",
        "Each taxonomy class against the simple feature rule that should fire on "
        "it. Low recall means the features miss the class; low precision means "
        "the rule also fires on other classes.", "",
        "| Class | Rule | n class | n fired | Recall | Precision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cls, d in ann["detectors"].items():
        rec = "n/a" if d["recall"] is None else f"{d['recall']:.0%}"
        pre = "n/a" if d["precision"] is None else f"{d['precision']:.0%}"
        lines.append(
            f"| {cls} | {d['meaning']} | {d['n_class']} | {d['n_fired']} | {rec} | {pre} |"
        )

    prof = ann["feature_means_by_type"]
    types = sorted({t for v in prof.values() for t in v})
    lines += [
        "", "## Feature means by class", "",
        "| Feature | " + " | ".join(types) + " |",
        "| --- |" + " --- |" * len(types),
    ]
    for feat, by_type in prof.items():
        cells = ["" if by_type.get(t) is None else f"{by_type[t]:.2f}" for t in types]
        lines.append(f"| `{feat}` | " + " | ".join(cells) + " |")
    lines += [
        "", "> The sample is small (tens per class). Treat these as direction "
        "checks, not performance estimates.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Feature dictionary
# --------------------------------------------------------------------------

DEFINITIONS: dict[str, str] = {
    "cust_stated_neither_date_nor_amount":
        "Borrower never uttered a date or an amount anywhere in the prefix. The "
        "core agent-recorded/agent-pushed detector: if this is 1 at end of call, "
        "whatever was logged as a promise was constructed by the house.",
    "first_date_speaker":
        "Who first mentioned any date: customer, house (bot/agent), or none.",
    "first_amount_speaker":
        "Who first mentioned a money amount. House-first suggests the number was "
        "pushed rather than offered.",
    "acct_days_promised_before_salary":
        "Days from the promised date forward to the salary credit day, computed "
        "circularly within the month. Small positive values (1-5) mean the "
        "promise falls just BEFORE payday and is structurally hard to keep.",
    "acct_promise_before_payday":
        "1 when the promise lands 1-5 days before salary credit. The zero-model "
        "infeasibility flag.",
    "acct_promise_to_ability_ratio":
        "Promised amount divided by (ability_to_pay_estimate x EMI). Above 1 "
        "means the borrower promised more than their estimated capacity.",
    "acct_promise_to_overdue_ratio":
        "Promised amount as a share of what is overdue. Below 1 means a part-"
        "payment promise, which is not in itself a negative signal.",
    "agent_hist_kept_rate":
        "Agent's historical PTP-to-payment rate, using ONLY promises whose "
        "outcome was knowable before this call (end of the grace window < now) and "
        "ONLY train-split accounts.",
    "agent_hist_ptp_per_connect":
        "Promises logged per connected call, over attempts that started strictly "
        "before this call started (the call's own attempt is excluded). An "
        "unusually high value is the first sign of agent-pushed promises.",
    "agent_late_shift_bunching":
        "Share of the agent's promises logged after 18:00, averaged over their "
        "PREVIOUS days only. Target-chasing behaviour.",
    "agent_remark_reuse_rate":
        "Share of the agent's remarks that duplicate their other remarks, over "
        "previous days only. Copy-paste remarks accompany promises never taken.",
    "pause_on_promise_z":
        "Pause before the promise turn, z-scored against the borrower's own "
        "pauses in this call, so it is comparable across fast and slow talkers.",
    "latency_to_date_question_ms":
        "How long the borrower took to answer the 'by which date' question -- "
        "the single most informative pause in the call.",
    "asr_conf_dip_on_promise":
        "Drop in ASR confidence on the promise turn versus the borrower's call "
        "average. A hesitation proxy, standing in for the audio we do not have.",
    "safety_signal_any":
        "1 if any hardship or refusal cue fired. A hit here must never be "
        "allowed to escalate treatment -- route to restructuring or a human.",
    "obs_lang_mix":
        "0 for monolingual speech, higher for more code-mixing. Derived from "
        "observed markers, not from the registered preferred_language.",
    "cust_counteroffered":
        "Borrower stated an amount different from the house figure in force at "
        "that moment (the most recent house amount before it). Evidence of real "
        "negotiation rather than passive agreement.",
    "cust_to_house_amount_ratio":
        "First borrower amount divided by the house amount in force just before it.",
    "acct_n_prior_ptps_in_data":
        "Promises on this account in ptps.csv captured strictly before this one. "
        "The running counterpart to the window-start snapshot prev_ptp_count.",
    "acct_promise_horizon_days":
        "Days from the capture date to the promised date.",
    "acct_is_part_payment_promise":
        "1 when the promised amount is below 99% of the overdue; <NA> when either "
        "is missing.",
    "acct_is_repeat_promiser":
        "1 when the account had at least 2 earlier promises and at least 2 broken "
        "(from the accounts.csv snapshot).",
    "agent_language_match":
        "1 when the agent's language team matches the borrower's registered "
        "language (team 'all' always matches); <NA> when either is missing.",
    "agent_hist_n_resolved":
        "Number of the agent's resolved train-pool promises behind agent_hist_kept_rate.",
    "agent_hist_n_connects":
        "Number of the agent's connected train-pool attempts behind "
        "agent_hist_ptp_per_connect.",
    "post_engagement_score":
        "Link opened + confirmation 'yes' + inbound call, each 0/1, counted only "
        "for events before the promised date.",
    "post_hours_to_link_open":
        "Hours from capture to the first link open, if it happened before the "
        "promised date.",
    "promise_follows_minimum_ask":
        "The promise turn came directly after the house pushed for a minimum or "
        "part payment -- i.e. the number may be the house's, not the borrower's.",
    "promise_position_ratio":
        "Where in the call the promise appeared, 0 (start) to 1 (end). Promises "
        "at the very end often accompany an attempt to exit the call.",
}


def build_feature_dictionary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for table_name, table in outputs.items():
        if table_name.startswith("labels"):
            continue
        for col in table.columns:
            rows.append(
                {
                    "table": table_name,
                    "feature": col,
                    "group": feature_group(col),
                    "availability": classify_feature(col),
                    "dtype": str(table[col].dtype),
                    "null_rate": round(float(table[col].isna().mean()), 4),
                    "n_unique": int(table[col].nunique(dropna=True)),
                    "is_categorical": col in CATEGORICAL_FEATURES,
                    "as_of_caveat": as_of_caveat(col),
                    "definition": DEFINITIONS.get(col, ""),
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["table", "group", "feature"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_reports(
    frames: dict[str, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
    cfg: Config = DEFAULT,
) -> dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # ---- coverage ---------------------------------------------------------
    cov = coverage(frames, outputs)
    path = cfg.out_dir / "coverage_report.md"
    lines = ["# Coverage report", ""]
    lines += [
        "## Reach",
        "",
        f"- Promises in ptps.csv: **{cov['n_ptps']:,}**",
        f"- With a usable transcript: **{cov['n_ptps_with_transcript']:,}** "
        f"({cov['pct_ptps_with_transcript']:.1%})",
        f"- Sourced from calls: {cov['n_ptps_from_calls']:,}",
        f"- Sourced from field visits (no transcript): {cov['n_ptps_from_visits']:,}",
        f"- Distinct calls in transcripts: {cov['n_distinct_calls_in_transcripts']:,}",
        f"- In-call decision points generated: **{cov['n_incall_decision_points']:,}** "
        f"(mean {cov['mean_decision_points_per_ptp']} per promise)",
        "",
        f"- Turns per call: min {cov['turns_per_call']['min']}, "
        f"median {cov['turns_per_call']['median']}, max {cov['turns_per_call']['max']}",
        "",
        "## Lexicon coverage (the Phase 2 objective)",
        "",
    ]
    overall = cov["lexicon_coverage_overall"]
    lines.append(
        f"Borrower utterances with no cue of any category: "
        f"**{overall.get('uncovered_rate', float('nan')):.1%}** "
        f"of {int(overall.get('n_utterances', 0)):,}"
    )
    lines += ["", "| Category | % of borrower utterances |", "| --- | --- |"]
    for key, val in sorted(overall.items()):
        if key.startswith("pct_utterances_"):
            lines.append(f"| {key.replace('pct_utterances_', '')} | {val:.2%} |")

    per_lang = cov.get("lexicon_coverage_by_registered_language", {})
    if per_lang:
        lines += ["", "### Uncovered rate by registered language", "",
                  "| Language | Uncovered | Utterances |", "| --- | --- | --- |"]
        for language, stats in sorted(per_lang.items()):
            lines.append(
                f"| {language} | {stats.get('uncovered_rate', float('nan')):.1%} "
                f"| {int(stats.get('n_utterances', 0)):,} |"
            )
        lines += [
            "",
            "> A high uncovered rate for one language means the lexicon has a hole "
            "there. Re-run `lexicons.mine_candidate_terms` on that subset.",
        ]

    lines += [
        "", "## Gaps", "",
        f"- Call-sourced promises whose call has no transcript rows: "
        f"**{cov['n_call_ptps_missing_transcript']}** (kept with has_transcript = 0)",
    ]

    lines += ["", "## Splits", ""]
    for split_name, n in cov["split_sizes"].items():
        lines.append(f"- {split_name}: {n:,} accounts")
    lines.append(
        f"- Accounts appearing in more than one split: "
        f"**{cov['accounts_in_multiple_splits']}** (must be 0)"
    )

    lines += ["", "## Highest null-rate features", "", "| Feature | Null rate |", "| --- | --- |"]
    for feat, rate in cov["highest_null_features"].items():
        lines.append(f"| {feat} | {rate:.1%} |")

    path.write_text("\n".join(lines), encoding="utf-8")
    written["coverage"] = path
    (cfg.out_dir / "coverage_report.json").write_text(
        json.dumps(cov, indent=2, default=str), encoding="utf-8"
    )

    # ---- leakage ----------------------------------------------------------
    as_of = test_accounts_as_of(frames)
    labels = outputs.get("labels", pd.DataFrame())
    canary = (
        leak_canary(outputs["features_endcall"], labels, cfg)
        if not labels.empty
        else []
    )

    path = cfg.out_dir / "leakage_audit.md"
    lines = [
        "# Leakage audit", "",
        "## 1. Structural guarantee", "",
        "Post-call signals live in `features_postcall` and nowhere else. "
        "`pipeline.run()` asserts this and raises if any `post_` column appears "
        "in the in-call or end-call tables, so the separation cannot be "
        "defeated by accident.", "",
        "## 2. Are accounts.csv history columns as-of-capture?", "",
    ]
    for key, val in as_of.items():
        if key == "verdict":
            continue
        lines.append(f"- `{key}`: {_fmt(val)}")
    if "verdict" in as_of:
        lines += ["", f"**Verdict:** {as_of['verdict']}"]

    lines += [
        "", "## 3. Leak canary", "",
        f"Conversational features correlating with the label above "
        f"|r| = {cfg.features.leak_canary_abs_corr} (train split only; label "
        f"source: `{_label_source(labels)}`):", "",
    ]
    if canary:
        lines += ["| Feature | Correlation |", "| --- | --- |"]
        lines += [f"| {name} | {corr:+.3f} |" for name, corr in canary]
        lines += ["", "> Investigate each. A strong legitimate feature can trip this."]
    else:
        lines.append("None. (Smoke alarm did not fire.)")

    lines += [
        "", "## 4. Agent features", "",
        f"- History pool: **{cfg.features.agent_history_pool}** accounts only",
        f"- Outcome-known-time guard: **{cfg.features.respect_outcome_known_time}**",
        f"- Minimum prior observations before a rate is emitted: "
        f"**{cfg.features.min_agent_history}**", "",
        f"- Outcome known at: end of day promised_date + {cfg.label.grace_days} "
        "(the same window the label uses)", "",
        "Agent rates use only promises already resolved before the current call. "
        "A promise due next week contributes nothing today, even though it sits "
        "earlier in the file. Promise-per-connect counts only attempts that "
        "started before this call started, so the call's own PTP never counts.", "",
        "## 5. Post-call table cut-off", "",
        f"`features_postcall` counts only events between capture and promised_date "
        f"+ {cfg.features.postcall_horizon_days} day(s). Events after the promise "
        "falls due mostly accompany the payment itself, so counting them would "
        "leak the outcome into the parking-window model.", "",
        "## 6. Labels and annotations never enter a feature table", "",
        "`pipeline.run()` raises if `kept`, `paid`, `amount_ratio`, `label_source`, "
        "`annotated_type` or `annotator_confidence` appears in any feature table.", "",
        "## 7. Known caveats for Team C", "",
        "Agent-identity features make the model partly predict *the agent* rather "
        "than *the promise*. The brief requires the scorer stay independent of "
        "the agents it measures. They are grouped under the `agent_` prefix so "
        "they can be ablated in one line -- please report performance with and "
        "without them.", "",
        "Columns built on `accounts.csv` fields with undocumented as-of semantics "
        "carry an `as_of_caveat` in `feature_dictionary.csv`. "
        "`acct_n_prior_ptps_in_data` is the safe running alternative to "
        "`acct_prev_ptp_count`.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    written["leakage"] = path

    # ---- dictionary -------------------------------------------------------
    dictionary = build_feature_dictionary(outputs)
    path = cfg.out_dir / "feature_dictionary.csv"
    dictionary.to_csv(path, index=False)
    written["dictionary"] = path

    path = cfg.out_dir / "feature_dictionary.md"
    lines = ["# Feature dictionary", "",
             "Generated from the code, so it cannot drift from what is shipped.", ""]
    for availability, grp in dictionary.groupby("availability"):
        lines += [f"## {availability}", "", f"_{AVAILABILITY.get(availability, '')}_", ""]
        lines += ["| Table | Feature | Group | Nulls | Caveat | Definition |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for row in grp.itertuples(index=False):
            lines.append(
                f"| {row.table} | `{row.feature}` | {row.group} | "
                f"{row.null_rate:.1%} | {row.as_of_caveat} | {row.definition} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    written["dictionary_md"] = path

    # ---- annotation agreement ---------------------------------------------
    ann = annotation_check(frames, outputs)
    if ann:
        path = cfg.out_dir / "annotation_check.md"
        path.write_text(_annotation_markdown(ann), encoding="utf-8")
        (cfg.out_dir / "annotation_check.json").write_text(
            json.dumps(ann, indent=2, default=str), encoding="utf-8"
        )
        written["annotations"] = path

    return written
