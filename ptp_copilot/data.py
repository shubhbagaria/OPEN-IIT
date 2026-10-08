"""Loading and light cleaning of the raw CreditNirvana extracts."""
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR


@lru_cache(maxsize=None)
def load_all():
    rd = lambda f, **kw: pd.read_csv(DATA_DIR / f, **kw)
    ptps = rd("ptps.csv", parse_dates=["captured_ts", "promised_date"])
    payments = rd("payments.csv", parse_dates=["payment_ts"])
    accounts = rd("accounts.csv")
    agents = rd("agents.csv")
    dials = rd("dial_attempts.csv", parse_dates=["attempt_ts"])
    transcripts = rd("call_transcripts.csv")
    events = rd("post_call_events.csv", parse_dates=["event_ts"])
    annotations = rd("annotated_ptp_sample.csv")
    splits = rd("splits.csv")
    lenders = rd("lenders.csv")

    transcripts["text"] = transcripts["text"].fillna("").astype(str)
    transcripts = transcripts.sort_values(["call_id", "turn_idx"]).reset_index(drop=True)
    ptps = ptps.merge(splits, on="account_id", how="left")
    ptps = ptps.sort_values("captured_ts").reset_index(drop=True)
    return dict(ptps=ptps, payments=payments, accounts=accounts, agents=agents, dials=dials,
                transcripts=transcripts, events=events, annotations=annotations,
                splits=splits, lenders=lenders)
