"""
Pipeline orchestration: raw CSVs in, three feature tables out.

THE THREE TABLES AND WHY THEY ARE SEPARATE
------------------------------------------
  features_endcall   ptp_id grain. Everything knowable by the moment the call
                     ends. This is Team C's main training table.

  features_incall    ptp_id x turn_idx grain. One row per customer turn, each
                     computed from that prefix only. This is what makes a
                     mid-call model possible, and what Phase 4 serves.

  features_postcall  ptp_id grain. Adds post_call_events (link opened, WhatsApp
                     reply, inbound callback). USABLE ONLY for the parking-window
                     and follow-up models.

The separation is physical, not a naming convention. post-call signals never
appear in the first two tables, so it is impossible to leak them into the
in-call model by accident. That is worth more than any amount of discipline.

GRAIN
-----
Everything is keyed on ptp_id, never call_id. ptps.csv carries both attempt_id
and visit_id, and field-sourced promises have no transcript at all. Those rows
are kept with has_transcript = 0 and NaN conversational features, because Team C
may well want them for the account-only model.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, DEFAULT, OPTIONAL_FILES
from .context import build_account_features, build_agent_features
from .features import (
    CATEGORICAL_FEATURES,
    extract_prefix_features,
    prepare_turns,
)

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

EXPECTED_COLUMNS: dict[str, tuple[str, ...]] = {
    "ptps": (
        "ptp_id", "account_id", "source", "attempt_id", "visit_id", "channel",
        "agent_id", "captured_ts", "promised_date", "promised_amount",
        "overdue_at_capture", "answerer_recorded", "payment_link_sent",
    ),
    "call_transcripts": (
        "call_id", "turn_idx", "speaker", "start_ms", "end_ms",
        "pause_before_ms", "text", "asr_confidence",
    ),
    "post_call_events": ("ptp_id", "event_type", "event_ts"),
    "accounts": ("account_id", "lender_id", "preferred_language"),
    "agents": ("agent_id", "channel", "language_team", "tenure_months", "shift"),
    "payments": ("payment_id", "account_id", "payment_ts", "amount", "channel"),
    "dial_attempts": (
        "attempt_id", "account_id", "attempt_ts", "agent_id",
        "talk_duration_s", "remark", "ptp_id",
    ),
    "splits": ("account_id", "split"),
    "lenders": ("lender_id", "lender_type"),
    # Optional (see config.OPTIONAL_FILES): validated only if present.
    "annotated_ptp_sample": ("ptp_id", "annotated_type", "annotator_confidence"),
}


def load_raw(cfg: Config = DEFAULT) -> dict[str, pd.DataFrame]:
    """Read every input CSV and fail loudly on a missing required column.

    Optional inputs (annotated_ptp_sample) are loaded when present and simply
    absent from the returned dict otherwise.
    """
    frames: dict[str, pd.DataFrame] = {}
    missing_files: list[str] = []

    for key in EXPECTED_COLUMNS:
        path = cfg.path(key)
        if not path.exists():
            if key not in OPTIONAL_FILES:
                missing_files.append(str(path))
            continue
        frames[key] = pd.read_csv(path)

    if missing_files:
        raise FileNotFoundError(
            "Missing input file(s):\n  " + "\n  ".join(missing_files)
            + f"\n\nSet PTP_DATA_DIR or pass --data-dir. Currently: {cfg.data_dir}"
        )

    problems: list[str] = []
    for key, required in EXPECTED_COLUMNS.items():
        if key not in frames:
            continue
        absent = [c for c in required if c not in frames[key].columns]
        if absent:
            problems.append(f"{key}.csv is missing columns: {absent}")
    if problems:
        raise ValueError("Schema check failed:\n  " + "\n  ".join(problems))

    return frames


# --------------------------------------------------------------------------
# Labels (only needed for agent history; Phase 1 owns the real thing)
# --------------------------------------------------------------------------


def load_or_build_labels(frames: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """
    Prefer Phase 1's label file. Fall back to a conservative recomputation.

    Team B needs a label for exactly one feature family -- agent historical
    keep rate. If Phase 1 has published labels, those are authoritative and we
    use them unchanged rather than maintaining a second definition.
    """
    for folder in (cfg.data_dir, cfg.out_dir):
        candidate = Path(folder) / cfg.label.phase1_label_file
        if candidate.exists():
            labels = pd.read_csv(candidate)
            if {"ptp_id", "kept"}.issubset(labels.columns):
                labels = labels[["ptp_id", "kept"]].copy()
                labels["label_source"] = f"phase1:{candidate.name}"
                return labels

    labels = _recompute_labels(frames, cfg)
    labels["label_source"] = "team_b_fallback"
    return labels


def _recompute_labels(frames: dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """
    Fallback label. Borrower-initiated payments only, inside the promise window.

    nach_represent is excluded on purpose: an auto-debit mandate clearing on
    retry is the plumbing working, not a borrower keeping a promise. Counting
    it would train a mandate-health model wearing a sincerity label.

    Each payment is credited to ONE promise: the most recently captured promise
    on the account whose window contains it. Without this, a payment landing in
    two overlapping windows (a re-promise made before the first fell due) would
    make both promises "kept".
    """
    lc = cfg.label
    ptps = frames["ptps"].copy()
    pay = frames["payments"].copy()
    if "payment_id" not in pay.columns:
        pay["payment_id"] = np.arange(len(pay))

    ptps["captured_ts"] = pd.to_datetime(ptps["captured_ts"], errors="coerce")
    ptps["promised_date"] = pd.to_datetime(ptps["promised_date"], errors="coerce")
    ptps["window_end"] = lc.window_end(ptps["promised_date"])  # exclusive

    pay["payment_ts"] = pd.to_datetime(pay["payment_ts"], errors="coerce")
    pay = pay[pay["channel"].isin(lc.borrower_initiated_channels)]

    merged = ptps[
        ["ptp_id", "account_id", "captured_ts", "window_end"]
    ].merge(pay[["payment_id", "account_id", "payment_ts", "amount"]], on="account_id")

    in_window = (merged["payment_ts"] >= merged["captured_ts"]) & (
        merged["payment_ts"] < merged["window_end"]
    )
    credited = (
        merged[in_window]
        .sort_values(["payment_id", "captured_ts", "ptp_id"])
        .drop_duplicates("payment_id", keep="last")
    )

    paid = credited.groupby("ptp_id", as_index=False)["amount"].sum().rename(
        columns={"amount": "paid"}
    )
    out = ptps[["ptp_id", "promised_amount"]].merge(paid, on="ptp_id", how="left")
    out["paid"] = out["paid"].fillna(0.0)

    ratio = out["paid"] / out["promised_amount"].replace(0, np.nan)
    out["kept"] = (ratio >= lc.kept_amount_ratio).astype(int)
    out["amount_ratio"] = ratio
    return out[["ptp_id", "kept", "amount_ratio", "paid"]]


# --------------------------------------------------------------------------
# Conversational tables
# --------------------------------------------------------------------------


def build_conversational_tables(
    frames: dict[str, pd.DataFrame], cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce (end-of-call table, in-call streaming table).

    Both come from the same `extract_prefix_features`. The end-of-call row is
    simply the final prefix -- there is no separate "aggregate the whole call"
    code path, by design.
    """
    fc = cfg.features
    ptps = frames["ptps"]
    transcripts = frames["call_transcripts"]

    # ptps.attempt_id joins to transcripts.call_id.
    link = ptps[["ptp_id", "attempt_id"]].dropna(subset=["attempt_id"])
    by_call = {
        call_id: group.to_dict("records")
        for call_id, group in transcripts.groupby("call_id", sort=False)
    }

    end_rows: list[dict] = []
    stream_rows: list[dict] = []

    for ptp_id, attempt_id in link.itertuples(index=False):
        rows = by_call.get(attempt_id)
        if not rows:
            continue

        records = prepare_turns(
            rows, min_bare_amount=fc.min_bare_number_as_amount
        )
        if not records:
            continue

        # --- end of call: the final prefix ---------------------------------
        end = extract_prefix_features(records, len(records) - 1)
        end.update({"ptp_id": ptp_id, "call_id": attempt_id, "has_transcript": 1})
        end_rows.append(end)

        # --- streaming: one row per customer turn --------------------------
        n_cust = 0
        for k, rec in enumerate(records):
            if rec.speaker != fc.emit_rows_on_speaker:
                continue
            n_cust += 1
            if n_cust < fc.min_customer_turns_for_decision:
                continue
            snap = extract_prefix_features(records, k)
            snap.update(
                {
                    "ptp_id": ptp_id,
                    "call_id": attempt_id,
                    "decision_point": n_cust,
                    "is_final_turn": int(k == len(records) - 1),
                }
            )
            stream_rows.append(snap)

    endcall = pd.DataFrame(end_rows)
    incall = pd.DataFrame(stream_rows)

    # Field-sourced promises have no call. Keep them, flagged.
    all_ptps = ptps[["ptp_id"]].copy()
    endcall = all_ptps.merge(endcall, on="ptp_id", how="left")
    endcall["has_transcript"] = endcall["has_transcript"].fillna(0).astype(int)

    return endcall, incall


# --------------------------------------------------------------------------
# Post-call table
# --------------------------------------------------------------------------

POST_CALL_EVENT_TYPES = (
    "payment_link_sent",
    "payment_link_opened",
    "whatsapp_confirmation_sent",
    "confirmation_reply_yes",
    "confirmation_reply_no",
    "inbound_call_from_customer",
)


def build_postcall_features(
    frames: dict[str, pd.DataFrame], cfg: Config
) -> pd.DataFrame:
    """
    Signals that arrive AFTER the call ends.

    Quarantined in their own table. A borrower opening the payment link is one
    of the strongest available predictors, which is exactly why it must never
    reach the in-call model -- at prompt time it has not happened yet.

    Legitimate consumers: the parking-window / hazard model, and next-day
    follow-up prioritisation.

    Only events in [captured_ts, promised_date + postcall_horizon_days) count.
    Events after the promise falls due mostly accompany the payment itself (a
    link opened in order to pay), so counting them would hand the outcome to
    the parking-window model.
    """
    horizon = pd.Timedelta(days=cfg.features.postcall_horizon_days)
    ptps = frames["ptps"][["ptp_id", "captured_ts", "promised_date"]].copy()
    ptps["captured_ts"] = pd.to_datetime(ptps["captured_ts"], errors="coerce")
    ptps["promised_date"] = pd.to_datetime(ptps["promised_date"], errors="coerce")

    ev = frames["post_call_events"].copy()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"], errors="coerce")
    ev = ev.merge(ptps, on="ptp_id", how="inner")
    ev = ev[
        (ev["event_ts"] >= ev["captured_ts"])
        & (ev["event_ts"] < ev["promised_date"] + horizon)
    ]

    out = ptps[["ptp_id"]].copy()

    for event in POST_CALL_EVENT_TYPES:
        sub = ev[ev["event_type"] == event]
        counts = sub.groupby("ptp_id").size().rename(f"post_{event}_n")
        first = sub.groupby("ptp_id")["event_ts"].min().rename(f"post_{event}_first_ts")
        out = out.merge(counts, on="ptp_id", how="left").merge(
            first, on="ptp_id", how="left"
        )
        out[f"post_{event}_n"] = out[f"post_{event}_n"].fillna(0).astype(int)
        out[f"post_{event}_any"] = (out[f"post_{event}_n"] > 0).astype(int)

    out = out.merge(ptps, on="ptp_id", how="left")

    # Hours from call end to first link open -- speed of engagement.
    opened = out.get("post_payment_link_opened_first_ts")
    if opened is not None:
        out["post_hours_to_link_open"] = (
            opened - out["captured_ts"]
        ).dt.total_seconds() / 3600.0

    out["post_engagement_score"] = (
        out.get("post_payment_link_opened_any", 0)
        + out.get("post_confirmation_reply_yes_any", 0)
        + out.get("post_inbound_call_from_customer_any", 0)
    )

    drop = ["captured_ts", "promised_date"] + [
        c for c in out.columns if c.endswith("_first_ts")
    ]
    return out.drop(columns=[c for c in drop if c in out.columns])


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def run(
    cfg: Config = DEFAULT, frames: dict[str, pd.DataFrame] | None = None
) -> dict[str, pd.DataFrame]:
    """Build everything and write it to cfg.out_dir.

    Pass `frames` from an earlier `load_raw` call to avoid reading the CSVs twice.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    if frames is None:
        frames = load_raw(cfg)
    labels = load_or_build_labels(frames, cfg)

    endcall_conv, incall_conv = build_conversational_tables(frames, cfg)
    account = build_account_features(
        frames["ptps"], frames["accounts"], frames["lenders"]
    )
    agent = build_agent_features(
        frames["ptps"],
        frames["dial_attempts"],
        frames["agents"],
        frames["accounts"],
        frames["splits"],
        labels,
        cfg,
    )
    postcall = build_postcall_features(frames, cfg)

    # --- end-of-call table -------------------------------------------------
    endcall = (
        endcall_conv.merge(account, on="ptp_id", how="left", validate="one_to_one")
        .merge(agent, on="ptp_id", how="left", validate="one_to_one")
        .merge(frames["splits"], on="account_id", how="left", validate="many_to_one")
    )

    # --- in-call table -----------------------------------------------------
    # Account and agent context are known before the call starts, so they are
    # legitimately available at every decision point.
    if not incall_conv.empty:
        incall = (
            incall_conv.merge(account, on="ptp_id", how="left")
            .merge(agent, on="ptp_id", how="left")
            .merge(frames["splits"], on="account_id", how="left")
        )
    else:
        incall = incall_conv

    # --- post-call table ---------------------------------------------------
    postcall_full = postcall.merge(
        endcall[["ptp_id", "account_id", "split"]], on="ptp_id", how="left"
    )

    # HARD GUARANTEE: no post-call column may exist in the first two tables.
    post_cols = {c for c in postcall.columns if c.startswith("post_")}
    for name, table in (("endcall", endcall), ("incall", incall)):
        intruders = post_cols.intersection(table.columns)
        if intruders:
            raise AssertionError(
                f"Post-call leakage into features_{name}: {sorted(intruders)}"
            )

    # Same guarantee for labels: the outcome and the human annotation must
    # never sit in a feature table.
    label_cols = {"kept", "amount_ratio", "paid", "label_source",
                  "annotated_type", "annotator_confidence"}
    for name, table in (("endcall", endcall), ("incall", incall), ("postcall", postcall_full)):
        intruders = label_cols.intersection(table.columns)
        if intruders:
            raise AssertionError(f"Label leakage into features_{name}: {sorted(intruders)}")

    outputs = {
        "features_endcall": endcall,
        "features_incall": incall,
        "features_postcall": postcall_full,
        # Whichever label the agent history used (Phase 1 if present, else the
        # Team B fallback); `label_source` says which.
        "labels": labels,
    }

    for name, table in outputs.items():
        _write(table, cfg.out_dir / name)

    return outputs


def _write(df: pd.DataFrame, stem: Path) -> None:
    """CSV always; Parquet too when an engine (pyarrow) is installed.

    A Parquet failure is reported, not swallowed, and any half-written file is
    removed so a stale or truncated .parquet is never left next to the CSV.
    """
    df.to_csv(stem.with_suffix(".csv"), index=False)
    target = stem.with_suffix(".parquet")
    try:
        df.to_parquet(target, index=False)
    except Exception as exc:  # missing engine or an unserialisable column
        target.unlink(missing_ok=True)
        warnings.warn(f"Parquet not written for {stem.name}: {exc}", stacklevel=2)
