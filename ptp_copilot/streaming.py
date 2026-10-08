"""Real-time in-call copilot (tabular track).

`InCallCopilot` is the object a telephony/voice-bot integration would hold per
live call. Each ASR-finalised utterance is pushed with `on_turn(...)`; after
every turn it returns the current credibility score, likely PTP category,
guard-railed next utterance and whether it has "heard enough" to act.
It reuses *exactly* the feature code used in training (no train/serve skew).
"""
import pickle
import time

import numpy as np
import pandas as pd

from . import features as F
from . import policy as P
from .config import CATEGORIES, MODEL_DIR


class InCallCopilot:
    def __init__(self, bundle_path=MODEL_DIR / "models.pkl"):
        with open(bundle_path, "rb") as fh:
            b = pickle.load(fh)
        self.models = b["models"]
        F._CAT_LEVELS.update(b["cat_levels"])
        self.in_cols = self.models["in_call"].features

    def start_call(self, call_id, context: dict):
        """context = pre-call account/PTP-term features (one row of the feature store)."""
        self.call_id, self.ctx = call_id, dict(context)
        self.ctx_row = pd.DataFrame([self.ctx])
        self.turns, self.history, self.sig = [], [], None
        return self

    def on_turn(self, speaker, text, start_ms, end_ms, pause_before_ms, asr_confidence):
        t0 = time.perf_counter()
        self.turns.append(dict(call_id=self.call_id, turn_idx=len(self.turns), speaker=speaker, start_ms=start_ms,
                               end_ms=end_ms, pause_before_ms=pause_before_ms, text=str(text),
                               asr_confidence=asr_confidence))
        # lexical signals only for the new turn (3-turn window gives it its local context)
        new = F.turn_signals(pd.DataFrame(self.turns[-3:])).iloc[[-1]]
        self.sig = new if self.sig is None else pd.concat([self.sig, new], ignore_index=True)
        pf = F.prefix_features(self.sig)
        row = pf.iloc[[-1]].reset_index(drop=True)
        row = pd.concat([row, self.ctx_row.drop(columns=[c for c in self.ctx_row if c in row])], axis=1)
        sal = row["salary_day_said"].fillna(row["salary_credit_day"]).to_numpy(float)
        row["discussed_day_vs_salary"] = F._days_from_salary(row["day_under_discussion"].to_numpy(float), sal)
        M = F.to_model_frame(row, self.in_cols)
        pfo = np.array([self.ctx.get("portfolio")])
        p_keep = float(self.models["in_call"].predict(M, pfo)[0])
        Pc = self.models["category"].proba(M)[0]
        cat_model = CATEGORIES[int(self.models["category"].decide(Pc[None])[0])]
        feats = row.iloc[0].to_dict()
        cat, rails = P.apply_guardrails(cat_model, feats)
        pdate = pd.Timestamp(self.ctx["promised_date"]).strftime("%d %b")
        if feats["n_c_commit"] + feats["n_a_propose_date"] + feats["n_c_third_party"] == 0 and cat != "third_party":
            utt = P.WAIT_SCRIPT.get(self.ctx.get("preferred_language"), P.WAIT_SCRIPT["english"])
        else:
            utt = P.guidance(cat, self.ctx.get("preferred_language"), feats, p_keep, self.ctx["promised_amount"],
                             pdate, self.ctx.get("salary_credit_day"))
        issues = P.compliance_check(utt, cat)
        prev = self.history[-1] if self.history else None
        stable = prev is not None and abs(prev["p_keep"] - p_keep) < 0.05 and prev["category"] == cat and Pc.max() >= 0.5
        event = (feats["n_c_commit"] + feats["n_a_record"] + feats["n_a_propose_date"] + feats["n_c_third_party"]) > 0
        out = dict(turn=len(self.turns) - 1, speaker=speaker, text=text, p_keep=round(p_keep, 3), category=cat,
                   category_model=cat_model, category_conf=round(float(Pc.max()), 3), guardrails=rails,
                   next_utterance=utt, compliance_issues=issues, heard_enough=bool(stable and event),
                   latency_ms=round((time.perf_counter() - t0) * 1000, 2))
        self.history.append(out)
        return out


def replay(copilot, base_row, transcript):
    copilot.start_call(base_row["attempt_id"], base_row)
    return [copilot.on_turn(r.speaker, r.text, r.start_ms, r.end_ms, r.pause_before_ms, r.asr_confidence)
            for r in transcript.itertuples(index=False)]
