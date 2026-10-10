"""Application services: a thin integration layer over the existing `ptp_copilot` package.

Nothing here re-implements modelling. It reuses:
  * ptp_copilot.pipeline.build      - data + feature store (same code that trained the models)
  * ptp_copilot.features            - per-turn signals and prefix features
  * outputs/models/models.pkl       - the trained LightGBM in-call keep + category models
  * ptp_copilot.policy              - guardrails, approved scripts, compliance filter, parking engine
  * ptp_copilot.slm                 - the fine-tuned Qwen2.5-0.5B (merged LoRA) loader, prompt and code-logit parser
and the stored evaluation artefacts in outputs/.
"""
import json
import os
import pickle
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from ptp_copilot import features as F
from ptp_copilot import lexicon as lx
from ptp_copilot import policy as P
from ptp_copilot.config import CATEGORIES, DATA_DIR, MODEL_DIR, OUT_DIR

APP_CACHE = OUT_DIR / "app_cache.pkl"
AUDIT_LOG = OUT_DIR / "app_audit.jsonl"


def _clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def records(df):
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


# =============================================================================== data
class DataStore:
    def __init__(self):
        self.ready = False
        self.error = None
        self.loaded_at = None

    def load(self):
        t0 = time.time()
        # Content fingerprint of the inputs (not file timestamps: a fresh git clone on a host
        # gives every file a new mtime, which would wrongly force a slow, memory-heavy rebuild).
        import hashlib
        h = hashlib.sha1()
        for p in (OUT_DIR / "enriched_ptp_predictions.csv", MODEL_DIR / "models.pkl", DATA_DIR / "ptps.csv"):
            b = p.read_bytes()
            h.update(b.replace(b"\r\n", b"\n") if p.suffix == ".csv" else b)   # same hash on Windows and Linux checkouts
        fingerprint = h.hexdigest()
        c = None
        if APP_CACHE.exists():
            with open(APP_CACHE, "rb") as fh:
                c = pickle.load(fh)
            if c.get("fingerprint") != fingerprint:
                c = None
        if c is not None:
            print("[data] using prepared cache outputs/app_cache.pkl")
        else:
            from ptp_copilot.pipeline import build
            D, lab, ts, pf, base, conv_cols = build()
            c = dict(base=base, transcripts=D["transcripts"], payments=D["payments"], accounts=D["accounts"],
                     dials=D["dials"][["attempt_id", "account_id", "attempt_ts", "channel", "agent_id",
                                       "network_response", "disposition", "remark", "ptp_id", "talk_duration_s"]],
                     agents=D["agents"], lenders=D["lenders"], annotations=D["annotations"],
                     fingerprint=fingerprint)
            with open(APP_CACHE, "wb") as fh:
                pickle.dump(c, fh)
        self.base = c["base"]
        self.tr = c["transcripts"]
        self.payments = c["payments"]
        self.accounts = c["accounts"]
        self.dials = c["dials"]
        self.agents = c["agents"]
        self.lenders = c["lenders"]
        self.annotations = c["annotations"]
        self.en = pd.read_csv(OUT_DIR / "enriched_ptp_predictions.csv", parse_dates=["promised_date"])
        self.dash = pd.read_csv(OUT_DIR / "agent_integrity_dashboard.csv")
        self.results = json.load(open(OUT_DIR / "results.json"))
        self.lat_tab = json.load(open(OUT_DIR / "latency_tabular.json"))
        self.lat_slm = json.load(open(OUT_DIR / "latency_slm.json"))
        self.slm_eval = json.load(open(OUT_DIR / "slm_eval.json", encoding="utf-8"))
        ge = OUT_DIR / "gemini_eval.json"
        self.gemini_eval = json.load(open(ge))["final"] if ge.exists() else None
        self.base_by_ptp = self.base.set_index("ptp_id")
        self.en_by_ptp = self.en.set_index("ptp_id")
        self.calls_by_id = {k: g.reset_index(drop=True) for k, g in self.tr.groupby("call_id")}
        # calls that carry a PTP and have a transcript = replayable calls
        b = self.base[self.base["attempt_id"].isin(self.calls_by_id)]
        dur = self.tr.groupby("call_id")["end_ms"].max() / 1000
        nt = self.tr.groupby("call_id").size()
        calls = b[["ptp_id", "attempt_id", "account_id", "agent_id", "channel", "split", "captured_ts", "dpd_start",
                   "outstanding", "overdue_at_capture", "promised_amount", "promised_date", "preferred_language",
                   "portfolio", "lender_id"]].copy()
        calls["duration_s"] = calls["attempt_id"].map(dur)
        calls["n_turns"] = calls["attempt_id"].map(nt)
        calls = calls.merge(self.en[["ptp_id", "predicted_category", "category_confidence", "confidence_score",
                                     "queue", "status", "kept", "ground_truth_category", "category_source"]],
                            on="ptp_id", how="left")
        self.calls = calls.rename(columns={"attempt_id": "call_id"})
        self.ready = True
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[data] ready in {time.time() - t0:.1f}s: {len(self.base)} PTPs, {len(self.calls)} replayable calls")


# =============================================================================== tabular model
class TabularModel:
    def __init__(self):
        with open(MODEL_DIR / "models.pkl", "rb") as fh:
            b = pickle.load(fh)
        self.models = b["models"]
        F._CAT_LEVELS.update(b["cat_levels"])
        self.in_cols = self.models["in_call"].features
        self.latencies = deque(maxlen=200)

    def score(self, prefix_turns: pd.DataFrame, ctx: dict):
        """Score a transcript prefix exactly like streaming.InCallCopilot.on_turn does."""
        t0 = time.perf_counter()
        ts = F.turn_signals(prefix_turns)
        pf = F.prefix_features(ts)
        row = pf.iloc[[-1]].reset_index(drop=True)
        ctx_row = pd.DataFrame([ctx])
        row = pd.concat([row, ctx_row.drop(columns=[c for c in ctx_row if c in row])], axis=1)
        sal = row["salary_day_said"].fillna(row["salary_credit_day"]).to_numpy(float)
        row["discussed_day_vs_salary"] = F._days_from_salary(row["day_under_discussion"].to_numpy(float), sal)
        M = F.to_model_frame(row, self.in_cols)
        pfo = np.array([ctx.get("portfolio")])
        p_keep = float(self.models["in_call"].predict(M, pfo)[0])
        Pc = self.models["category"].proba(M)[0]
        cat = CATEGORIES[int(self.models["category"].decide(Pc[None])[0])]
        shap = self.models["in_call"].shap(M).iloc[0]
        ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(ms)
        return dict(p_keep=p_keep, probs=Pc, category=cat, feats=row.iloc[0].to_dict(), signals=ts,
                    shap=shap, M=M, latency_ms=ms)


# =============================================================================== Qwen SLM
class QwenService:
    """Loads the fine-tuned Qwen ONCE (background thread) through ptp_copilot.slm.load."""

    def __init__(self, precision=None):
        self.precision = precision or os.environ.get("PTP_SLM_PRECISION", "nf4")
        self.status = "not_loaded"
        self.error = None
        self.device = None
        self.vram_gb = None
        self.lock = threading.Lock()
        self.latencies = deque(maxlen=200)
        self.gen_latencies = deque(maxlen=50)
        self.loaded_at = None
        self.temperature = None

    def start(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        self.status = "loading"
        try:
            from ptp_copilot import slm
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU not available to torch")
            self.slm = slm
            self.tok, self.model, self.code_ids = slm.load(self.precision)
            self.device = torch.cuda.get_device_name(0)
            self.vram_gb = round(torch.cuda.memory_allocated() / 1e9, 3)
            # temperature fitted on validation for this precision (stored by the evaluation scripts)
            if self.precision == "nf4":
                self.temperature = float(json.load(open(OUT_DIR / "latency_slm.json"))
                                         ["nf4_quality_test_final"]["temperature"])
            else:
                self.temperature = float(json.load(open(OUT_DIR / "slm_eval.json", encoding="utf-8"))["temperature"])
            # warm-up (first CUDA call is slow)
            slm.code_logits(self.tok, self.model, self.code_ids, [slm.HEAD + "CTX\nCALL:\nC: ok\n" + slm.TAIL])
            self.status = "ready"
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[qwen] ready ({self.precision}) on {self.device}, {self.vram_gb} GB")
        except Exception as e:  # surfaced to the UI; never replaced by fake output
            self.status = "unavailable"
            self.error = f"{type(e).__name__}: {e}"
            print("[qwen] unavailable:", self.error)

    def prompt(self, ctx, turns):
        body = "".join(self.slm.turn_line(s, x) for s, x in turns)
        return self.slm.HEAD + self.slm.ctx_line(ctx) + body + self.slm.TAIL

    def classify(self, ctx, turns):
        if self.status != "ready":
            return dict(available=False, status=self.status, reason=self.error or "Qwen SLM is not loaded yet")
        import torch
        p = self.prompt(ctx, turns)
        with self.lock:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            lg = self.slm.code_logits(self.tok, self.model, self.code_ids, [p])
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
        self.latencies.append(ms)
        probs = self.slm.softmax(lg, self.temperature)[0]
        i = int(probs.argmax())
        return dict(available=True, category=CATEGORIES[i], confidence=float(probs[i]),
                    probs={c: float(v) for c, v in zip(CATEGORIES, probs)}, latency_ms=round(ms, 1),
                    prompt_tokens=len(self.tok(p, add_special_tokens=False)["input_ids"]))

    def generate(self, ctx, turns, max_new_tokens=90):
        if self.status != "ready":
            return dict(available=False, status=self.status, reason=self.error or "Qwen SLM is not loaded yet")
        import torch
        p = self.prompt(ctx, turns)
        with self.lock:
            enc = self.tok(p, return_tensors="pt", add_special_tokens=False).to("cuda")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
        new = g[0, enc["input_ids"].shape[1]:]
        txt = self.tok.decode(new, skip_special_tokens=True).strip()
        self.gen_latencies.append(ms)
        parts = txt.split("|", 2)
        ok = len(parts) == 3 and parts[0].strip() in self.slm.CODES
        return dict(available=True, raw=txt, format_valid=ok,
                    category=CATEGORIES[int(parts[0]) - 1] if ok else None,
                    utterance=parts[2].strip() if ok else None, tokens=int(len(new)), latency_ms=round(ms, 1))

    def version(self):
        log = {}
        try:
            log = json.load(open(OUT_DIR / "slm_train_log.json"))
        except Exception:
            pass
        mt = (OUT_DIR / "slm_merged" / "model.safetensors")
        built = time.strftime("%Y-%m-%d", time.localtime(mt.stat().st_mtime)) if mt.exists() else None
        return dict(name="Qwen2.5-0.5B-Instruct + LoRA (r=16, merged)", precision=self.precision,
                    trained_on=built, epochs=log.get("epochs"), steps=log.get("steps"),
                    version_tag=f"ptp-qwen0.5b-lora-r16-{built}-{self.precision}")


# =============================================================================== Gemini (default LLM)
class GeminiService:
    """Google Gemini via API (ptp_copilot.gemini_llm), same interface as QwenService.
    It replaces Qwen as the conversation model; the Qwen code is kept and selectable."""
    label = "Gemini"
    provider = "gemini"

    def __init__(self):
        from ptp_copilot.gemini_llm import MODEL
        self.model = MODEL
        self.precision = "api"
        self.status = "not_loaded"
        self.error = None
        self.device = "Google Cloud API"
        self.vram_gb = None
        self.latencies = deque(maxlen=200)
        self.gen_latencies = deque(maxlen=50)
        self.loaded_at = None
        self.temperature = None
        self.lock = threading.Semaphore(4)            # cap concurrent API calls from this server

    def start(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        self.status = "loading"
        try:
            from ptp_copilot.gemini_llm import GeminiClassifier, call_text
            self.clf = GeminiClassifier(self.model)
            self.call_text = call_text
            self.temperature = self.clf.T
            self.clf.classify_text("CTX lang=english portfolio=pl_salaried prior_broken=0 salary_day=? promise=Rs1000 on 01 Jan\n"
                                   "CALL:\nA: Hello, am I speaking with the borrower?\nC: Yes.")   # validates key + model
            self.status = "ready"
            self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[gemini] ready ({self.model}), temperature {self.temperature}")
        except Exception as e:  # surfaced to the UI; never replaced by fake output
            self.status = "unavailable"
            self.error = f"{type(e).__name__}: {str(e)[:300]}"
            print("[gemini] unavailable:", self.error)

    def _text(self, ctx, turns):
        from ptp_copilot.gemini_llm import ctx_line
        return self.call_text(ctx_line(ctx), turns)

    def classify(self, ctx, turns):
        if self.status != "ready":
            return dict(available=False, status=self.status, reason=self.error or "Gemini is not ready yet")
        try:
            with self.lock:
                o = self.clf.classify_text(self._text(ctx, turns), retries=3)
        except Exception as e:
            return dict(available=False, status="error", reason=f"{type(e).__name__}: {str(e)[:200]}")
        self.latencies.append(o["latency_ms"])
        i = int(o["probs"].argmax())
        return dict(available=True, category=CATEGORIES[i], confidence=float(o["probs"][i]),
                    probs={c: float(v) for c, v in zip(CATEGORIES, o["probs"])}, reasons=o["reasons"],
                    latency_ms=round(o["latency_ms"], 1), prompt_tokens=o["prompt_tokens"])

    def generate(self, ctx, turns, max_new_tokens=None):
        if self.status != "ready":
            return dict(available=False, status=self.status, reason=self.error or "Gemini is not ready yet")
        try:
            with self.lock:
                d = self.clf.draft(self._text(ctx, turns))
        except Exception as e:
            return dict(available=False, status="error", reason=f"{type(e).__name__}: {str(e)[:200]}")
        self.gen_latencies.append(d["latency_ms"])
        return dict(available=True, raw=d["text"], format_valid=True, category=None, utterance=d["text"],
                    tokens=None, latency_ms=round(d["latency_ms"], 1))

    def version(self):
        return dict(name=f"Google {self.model} (API, prompted, not fine-tuned)", precision="api",
                    trained_on=None, epochs=None, steps=None, version_tag=f"{self.model}-api")


def _cuda_available():
    try:
        import ptp_copilot.slm  # noqa: F401  (appends PTP_TORCH_SITE to sys.path if set)
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def make_llm():
    """Which conversation model sits next to LightGBM.

    Qwen (fine-tuned, local GPU) is THE secondary model of the project: every reported result,
    the evaluation, the report and the deck use it. Gemini (Google API) exists only so the
    website can be deployed on a host without a GPU, avoiding paid GPU hosting for Qwen.

      PTP_LLM_PROVIDER=auto   (default) Qwen if a CUDA GPU is available, otherwise Gemini
      PTP_LLM_PROVIDER=qwen   force the local Qwen SLM
      PTP_LLM_PROVIDER=gemini force Gemini (deployment); needs GEMINI_API_KEY
    """
    choice = os.environ.get("PTP_LLM_PROVIDER", "auto").lower()
    if choice == "auto":
        choice = "qwen" if _cuda_available() else "gemini"
    if choice == "qwen":
        q = QwenService()
        q.label, q.provider = "Qwen SLM", "qwen"
        return q
    return GeminiService()


# =============================================================================== assessment
EVIDENCE_RULES = [
    # (signal column in turn_signals, polarity, text)
    ("c_date", "+", "Customer stated a payment date"),
    ("c_amount", "+", "Customer stated an amount"),
    ("c_partial_offer", "+", "Customer offered a part payment"),
    ("c_firm", "+", "Firm commitment language from customer"),
    ("c_link_accept", "+", "Customer asked for / accepted the payment link"),
    ("c_hedge", "-", "Customer hedged ('dekhta hoon', 'try karunga', 'let me see')"),
    ("c_exit", "-", "Customer tried to end the call"),
    ("c_forgot", "-", "Customer said they forgot"),
    ("c_vague_date", "-", "Vague timing ('next week', 'agle hafte')"),
    ("c_conditional", "-", "Conditional promise ('salary aayi toh')"),
    ("c_link_refuse", "-", "Customer declined the payment link"),
    ("c_refusal", "-", "Customer said it is not possible now"),
    ("c_prior_break_excuse", "-", "Excuse for a previously broken promise"),
    ("c_ack_to_proposal", "-", "Only a minimal 'haan/ok' to the agent's proposed date"),
    ("a_pressure", "-", "Agent used pressure phrases (CIBIL / escalation / last reminder)"),
    ("a_propose_date", "~", "Agent proposed the payment date"),
    ("a_prior_break", "~", "Agent cited a previously broken promise"),
    ("c_hardship", "!", "Hardship mentioned by customer"),
    ("c_severe_hardship", "!", "Severe hardship (medical / job loss)"),
    ("c_third_party", "!", "Someone other than the borrower is on the line"),
    ("c_wrong_number", "!", "Wrong number indicated"),
]


def evidence(signals: pd.DataFrame, feats: dict, ctx: dict):
    out = []
    for col, pol, text in EVIDENCE_RULES:
        if col in signals and signals[col].sum() > 0:
            idx = signals.index[signals[col] > 0]
            t = signals.loc[idx[0]]
            out.append(dict(signal=col, polarity=pol, text=text, count=int(signals[col].sum()),
                            first_turn=int(t["turn_idx"]), quote=str(t["text"])[:160]))
    if feats.get("n_long_hesitation", 0):
        out.append(dict(signal="long_hesitation", polarity="-", text="Long hesitation (>1.2 s) before replying",
                        count=int(feats["n_long_hesitation"]), first_turn=None, quote=None))
    tpb = ctx.get("total_prior_broken") or 0
    if tpb >= 2:
        out.append(dict(signal="history", polarity="-", text=f"{int(tpb)} previous PTPs were not kept",
                        count=int(tpb), first_turn=None, quote=None))
    if ctx.get("is_salary_window"):
        out.append(dict(signal="salary_window", polarity="+", text="Promised date falls 0-5 days after salary credit",
                        count=1, first_turn=None, quote=None))
    if ctx.get("is_pre_salary_hazard"):
        out.append(dict(signal="pre_salary", polarity="-", text="Promised date falls 1-4 days before salary credit",
                        count=1, first_turn=None, quote=None))
    if (ctx.get("hardship_30d") or 0) > 0:
        out.append(dict(signal="hardship_30d", polarity="!", text="Hardship disposition on this account in the last 30 days",
                        count=int(ctx["hardship_30d"]), first_turn=None, quote=None))
    return out


def commitment(turns):
    """Who first stated the amount / date, read from the transcript with the existing lexicon."""
    first = {"date": None, "amount": None}
    last = {"date": None, "amount": None}
    cust_confirmed = False
    for i, (s, x) in enumerate(turns):
        txt = lx.clean(x)
        who = "customer" if s == "customer" else "agent"
        prior = bool(lx.AGENT_PRIOR_BREAK.search(txt)) and who == "agent"
        days = [] if (prior or lx.extract_salary_day(txt)) else lx.extract_days(txt)
        rel = bool(lx.DATE_RELATIVE.search(txt)) and not prior
        amts = lx.extract_amounts(txt)
        if days or rel:
            v = (f"day {days[-1]}" if days else lx.DATE_RELATIVE.search(txt).group(0))
            first["date"] = first["date"] or dict(by=who, value=v, turn=i)
            last["date"] = dict(by=who, value=v, turn=i)
        if amts:
            first["amount"] = first["amount"] or dict(by=who, value=amts[-1], turn=i)
            last["amount"] = dict(by=who, value=amts[-1], turn=i)
        if who == "customer" and (lx.FIRM.search(txt) or lx.PARTIAL_OFFER.search(txt)):
            cust_confirmed = True
    d_by = first["date"]["by"] if first["date"] else None
    a_by = first["amount"]["by"] if first["amount"] else None
    if d_by is None and a_by is None:
        own = "none_yet"
    elif d_by == "customer" or (a_by == "customer" and d_by != "agent"):
        own = "borrower_originated"
    elif cust_confirmed:
        own = "agent_proposed_borrower_confirmed"
    else:
        own = "agent_proposed"
    return dict(first=first, latest=last, borrower_explicit_confirmation=cust_confirmed, ownership=own)


EXTRA_LABELS = {
    "n_c_severe_hardship": "severe hardship mentioned", "cust_pause_max": "longest customer pause",
    "promised_dom": "promised day of month", "firmness_score": "borrower firmness score",
    "hedge_exit_score": "hedging / exit score", "n_c_amount": "customer stated an amount",
    "amount_to_emi": "promise size vs EMI", "salary_credit_day": "salary credit day", "income_type": "income type",
    "pause_reply_to_proposal_max": "pause before replying to agent's date", "cust_asr_min": "lowest ASR confidence",
    "talk_listen_ratio": "talk-to-listen ratio", "n_turns": "turns so far", "elapsed_s": "seconds into call",
    "agent_ptps_per_day": "agent PTPs per day", "capture_hour": "hour of capture", "channel": "channel",
}


def ask_points(cat, utt_kind, ctx, feats):
    """Key asks for the agent, as short chips. Uses the same numbers as policy.guidance
    (token amount, promised date, salary day + 2), so chips and the full sentence always agree."""
    if utt_kind.startswith("coaching"):
        return [dict(tag="ASK", text="Borrower's own payment date", tone="pay"),
                dict(tag="AVOID", text="Proposing a date yourself", tone="warn")]
    amt = float(ctx["promised_amount"])
    tok = P.token_amount(amt)
    rest = max(amt - tok, 0)
    pdate = pd.Timestamp(ctx["promised_date"]).strftime("%d %b")
    sd = feats.get("salary_day_said")
    sd = sd if sd is not None and not (isinstance(sd, float) and np.isnan(sd)) else ctx.get("salary_credit_day")
    alt = P.post_salary_day(sd)
    alt_txt = f"{P._ordinal(alt)} (salary + 2 days)" if alt else "2 days after salary"
    inr = P._inr
    return {
        "genuine_feasible": [dict(tag="CONFIRM", text=f"₹{inr(amt)} by {pdate}", tone="pay"),
                             dict(tag="SEND", text="Payment link on WhatsApp now"),
                             dict(tag="REMIND", text="1 day before the due date")],
        "genuine_infeasible": [dict(tag="OPTION A", text=f"Move date to {alt_txt}", tone="pay"),
                               dict(tag="OPTION B", text=f"₹{inr(tok)} today, rest later", tone="pay"),
                               dict(tag="OFFER", text="Restructuring check"),
                               dict(tag="AVOID", text="Pressure or escalation talk", tone="warn")],
        "escape": [dict(tag="PAY NOW", text=f"₹{inr(tok)} via payment link", tone="pay"),
                   dict(tag="REST", text=f"₹{inr(rest)} by {pdate}", tone="pay"),
                   dict(tag="FOLLOW UP", text="In 24–48 hours")],
        "third_party": [dict(tag="DO NOT", text="Mention the loan, amount or dues", tone="warn"),
                        dict(tag="ASK", text="Best time to reach the borrower", tone="pay"),
                        dict(tag="NEXT", text="Call the borrower directly within 24 h")],
        "repeat_promiser": [dict(tag="PAY NOW", text=f"₹{inr(tok)} part-payment today", tone="pay"),
                            dict(tag="REST", text=f"₹{inr(rest)} by {pdate}", tone="pay"),
                            dict(tag="STATE", text="Account may go to senior recovery review")],
        "agent_recorded_or_pushed": [dict(tag="CONFIRM", text=f"Borrower's own yes to ₹{inr(amt)} by {pdate}", tone="pay"),
                                     dict(tag="IF NOT", text="Ask the date that works for them"),
                                     dict(tag="FLAG", text="Integrity review")],
    }.get(cat, [])


def assess(store, tab, llm, ptp_id, upto_turn=None, use_llm=True):
    b = store.base_by_ptp.loc[ptp_id]
    ctx = b.to_dict()
    ctx["ptp_id"] = ptp_id
    call = store.calls_by_id.get(b["attempt_id"])
    if call is None:
        raise KeyError("no transcript for this PTP")
    n = len(call)
    k = n - 1 if upto_turn is None else max(0, min(int(upto_turn), n - 1))
    prefix = call.iloc[:k + 1]
    s = tab.score(prefix, ctx)
    feats = s["feats"]
    rail_feats = dict(feats)
    rail_feats["hardship_30d"] = ctx.get("hardship_30d", 0)
    final_cat, rails = P.apply_guardrails(s["category"], rail_feats)
    pdate = pd.Timestamp(ctx["promised_date"]).strftime("%d %b")
    committed = feats["n_c_commit"] + feats["n_a_propose_date"] + feats["n_c_third_party"] > 0
    lang = ctx.get("preferred_language")
    if not committed and final_cat != "third_party":
        utt = P.WAIT_SCRIPT.get(lang, P.WAIT_SCRIPT["english"])
        utt_kind = "coaching (no commitment yet)"
    else:
        utt = P.guidance(final_cat, lang, rail_feats, s["p_keep"], ctx["promised_amount"], pdate,
                         ctx.get("salary_credit_day"))
        utt_kind = "approved script"
    issues = P.compliance_check(utt, final_cat)
    park, rem, queue, fu = P.parking_plan(final_cat, s["p_keep"], ctx["promise_lag_days"], rails)
    turns = list(zip(prefix["speaker"], prefix["text"]))
    q = llm.classify(ctx, turns) if use_llm else dict(available=False, status="skipped", reason="not requested")
    q["label"] = getattr(llm, "label", "Qwen SLM")
    shap = s["shap"]
    top = shap.abs().sort_values(ascending=False).head(5).index
    from ptp_copilot.pipeline import READABLE
    drivers = [dict(feature=c, label=READABLE.get(c) or EXTRA_LABELS.get(c) or c.replace("n_c_", "customer ").replace("n_a_", "agent ").replace("_", " "), contribution=round(float(shap[c]), 4),
                    value=_clean(s["M"].iloc[0][c]) if not isinstance(s["M"].iloc[0][c], str) else s["M"].iloc[0][c])
               for c in top]
    hardship = (feats.get("n_c_hardship", 0) or 0) > 0 or (ctx.get("hardship_30d") or 0) > 0
    privacy = (feats.get("n_c_third_party", 0) or 0) > 0 or (feats.get("n_c_wrong_number", 0) or 0) > 0
    risk = "high" if s["p_keep"] < 0.25 else ("medium" if s["p_keep"] < 0.5 else "low")
    return dict(
        ptp_id=ptp_id, call_id=b["attempt_id"], account_id=ctx["account_id"], turn=k, n_turns=n,
        is_final=k == n - 1,
        credibility=dict(p_keep=round(s["p_keep"], 4), risk=risk,
                         source="LightGBM in-call keep model (calibrated)", status="MODEL OUTPUT"),
        tabular_category=dict(category=s["category"], confidence=round(float(s["probs"].max()), 4),
                              probs={c: round(float(v), 4) for c, v in zip(CATEGORIES, s["probs"])},
                              decision="cost-sensitive (hardship errors weighted 8x)"),
        llm=q,
        final_category=final_cat, guardrails=rails,
        hardship_safeguard=bool(hardship or any("hardship" in r or "vulnerable" in r for r in rails)),
        privacy_safeguard=bool(privacy),
        heard_enough=None,
        # same event definition as the offline "heard-enough" trigger (pipeline.streaming_eval)
        commitment_event=bool(feats["n_c_commit"] + feats["n_a_record"] + feats["n_a_propose_date"]
                              + feats["n_c_third_party"] > 0),
        commitment=commitment(turns),
        evidence=evidence(s["signals"], feats, ctx),
        drivers=drivers,
        treatment=dict(next_utterance=utt, utterance_kind=utt_kind, compliance_issues=issues,
                       ask=ask_points(final_cat, utt_kind, ctx, rail_feats),
                       parking_days=park, reminders=rem, queue=queue, follow_up=fu,
                       policy_check="pass" if not issues else "blocked"),
        latency_ms=dict(tabular=round(s["latency_ms"], 1), llm=q.get("latency_ms")),
        ground_truth=dict(category=_clean(store.en_by_ptp.loc[ptp_id, "ground_truth_category"]),
                          outcome=_clean(store.en_by_ptp.loc[ptp_id, "status"]),
                          note="Revealed for evaluation only; never used by the live models"),
    )


# =============================================================================== audit
class AuditLog:
    def __init__(self):
        self.lock = threading.Lock()

    def write(self, kind, a, model_version, extra=None):
        rec = dict(audit_id=uuid.uuid4().hex[:12], timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), kind=kind,
                   ptp_id=a["ptp_id"], call_id=a["call_id"], account_id=a["account_id"], turn=a["turn"],
                   n_turns=a["n_turns"], model_version=model_version,
                   p_keep=a["credibility"]["p_keep"], tabular_category=a["tabular_category"]["category"],
                   tabular_confidence=a["tabular_category"]["confidence"],
                   llm_label=a["llm"].get("label"), llm_category=a["llm"].get("category"),
                   llm_confidence=a["llm"].get("confidence"), llm_available=a["llm"].get("available"),
                   final_category=a["final_category"],
                   guardrails=a["guardrails"], hardship_safeguard=a["hardship_safeguard"],
                   privacy_safeguard=a["privacy_safeguard"], evidence=[e["text"] for e in a["evidence"]],
                   treatment=a["treatment"], policy_check=a["treatment"]["policy_check"],
                   audit_status="flagged" if (a["treatment"]["compliance_issues"] or a["guardrails"]) else "clear",
                   latency_ms=a["latency_ms"], extra=extra or {})
        with self.lock, open(AUDIT_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, default=_clean) + "\n")
        return rec

    def read(self):
        if not AUDIT_LOG.exists():
            return []
        with open(AUDIT_LOG, encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
