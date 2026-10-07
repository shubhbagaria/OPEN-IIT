"""
Phase 2 / Team B - configuration.

Every tunable number in the pipeline lives here so that nothing is buried as a
magic constant inside feature code. If a reviewer asks "why 3 days?", the answer
is a single line in this file with a comment next to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Root directory holding the raw CSVs. Override with the PTP_DATA_DIR env var
# or by passing --data-dir on the command line.
DATA_DIR = Path(os.environ.get("PTP_DATA_DIR", "csv_folder"))

# Where the feature store and reports are written.
OUT_DIR = Path(os.environ.get("PTP_OUT_DIR", "out"))

# The input files. Keys are the logical names used throughout the code. Nine
# are required; annotated_ptp_sample is optional (see OPTIONAL_FILES).
FILES = {
    # ps1_fake_ptp/
    "ptps": "ptps.csv",
    "call_transcripts": "call_transcripts.csv",
    "post_call_events": "post_call_events.csv",
    "annotated_ptp_sample": "annotated_ptp_sample.csv",
    # shared/
    "accounts": "accounts.csv",
    "agents": "agents.csv",
    "payments": "payments.csv",
    "dial_attempts": "dial_attempts.csv",
    "splits": "splits.csv",
    "lenders": "lenders.csv",
}

# Inputs the pipeline runs without. annotated_ptp_sample is a human-labelled
# taxonomy sample (ptp_id, annotated_type, annotator_confidence). It is used
# ONLY by the audit to validate features against human judgement -- it is a
# label, so it never enters a feature table.
OPTIONAL_FILES = ("annotated_ptp_sample",)


# --------------------------------------------------------------------------
# Label construction (needed only for agent as-of features; Phase 1 owns the
# authoritative label, and if a Phase 1 label file is present we use that
# instead of recomputing it here).
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelConfig:
    """
    Phase 1 owns the real label. Team B needs *a* label for exactly one purpose:
    computing an agent's historical PTP-to-payment rate. We therefore recompute a
    conservative version here, and we prefer the Phase 1 file whenever it exists.
    """

    # Payment channels that represent the borrower actively choosing to pay.
    # nach_represent is deliberately excluded: that is an auto-debit mandate
    # firing on retry, not a person keeping a promise.
    # upi_link is the borrower paying through the link sent on the call (the
    # single largest channel in payments.csv) and cash_field is cash handed to a
    # field agent; both are the borrower acting, so both count.
    borrower_initiated_channels: tuple[str, ...] = (
        "upi_app",
        "upi_link",
        "bbps",
        "neft",
        "branch_cash",
        "cash_field",
    )

    # Payments landing up to this many days after the promised date still count,
    # INCLUSIVE of the whole last day: grace_days=3 on a 3 April promise accepts
    # payments up to 23:59:59 on 6 April. Use `window_end()` rather than adding
    # the timedelta by hand, so the label and the agent as-of guard agree.
    grace_days: int = 3

    # Fraction of the promised amount that must be paid inside the window.
    kept_amount_ratio: float = 0.80

    # Filename Phase 1 writes its labels to. If present in DATA_DIR or OUT_DIR,
    # it wins over anything computed here. Expected columns: ptp_id, kept.
    phase1_label_file: str = "phase1_labels.csv"

    def window_end(self, promised_date):
        """Exclusive end of the payment window: midnight after the last grace day.

        Also the moment a promise's outcome becomes knowable, which is what the
        agent as-of features key on.
        """
        import pandas as pd

        return promised_date + pd.Timedelta(days=self.grace_days + 1)


# --------------------------------------------------------------------------
# Feature engineering behaviour
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    # --- streaming table ---------------------------------------------------
    # The in-call table emits one row per decision point. A decision point is a
    # customer turn: the model can only act after the borrower has said
    # something new. Emitting rows after bot turns would inflate the table
    # without adding information.
    emit_rows_on_speaker: str = "customer"

    # Don't emit a decision point before this many customer turns have happened.
    # Before that there is essentially nothing to score.
    min_customer_turns_for_decision: int = 1

    # --- agent context -----------------------------------------------------
    # Which accounts may contribute to historical agent statistics.
    # "train"  -> only train-split accounts (conservative, prevents any
    #             cross-split information flow; this is the default)
    # "all"    -> every account, still strictly backwards in time
    agent_history_pool: str = "train"

    # An agent's PTP-to-payment rate may only use promises whose outcome was
    # already *knowable* at the time of the current call. A promise due
    # tomorrow tells you nothing today. We treat an outcome as known at
    # the end of the grace window (LabelConfig.window_end).
    respect_outcome_known_time: bool = True

    # Minimum number of prior observations before an agent rate is emitted;
    # below this we emit NaN rather than a rate built on three calls.
    min_agent_history: int = 20

    # --- amount detection --------------------------------------------------
    # A bare number is only treated as a money amount if it is at least this
    # large or carries a currency marker / thousands separator. This stops
    # "7 dinadinda" (7 days) and "2 loans" from being read as rupee amounts.
    min_bare_number_as_amount: int = 100

    # --- post-call table ---------------------------------------------------
    # Post-call events are counted only up to this many days after the
    # promised date (exclusive, from midnight of the promised date). With 0,
    # only events before the promise falls due count. Events after that point
    # are usually the borrower paying or failing to, so including them leaks
    # the outcome into the parking-window model.
    postcall_horizon_days: int = 0

    # --- audit -------------------------------------------------------------
    # Any in-call feature correlating with the label above this absolute value
    # is flagged by the leak canary for manual review. It is a smoke alarm, not
    # a hard rule: a genuinely strong feature can trip it.
    leak_canary_abs_corr: float = 0.60

    # Features with a null rate above this are flagged in the coverage report.
    high_null_rate: float = 0.50


@dataclass(frozen=True)
class Config:
    data_dir: Path = DATA_DIR
    out_dir: Path = OUT_DIR
    label: LabelConfig = field(default_factory=LabelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    def path(self, key: str) -> Path:
        """Absolute path to one of the raw input files."""
        if key not in FILES:
            raise KeyError(f"Unknown input file key: {key!r}. Known: {sorted(FILES)}")
        return self.data_dir / FILES[key]


DEFAULT = Config()
