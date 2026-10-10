"""Gemini track: Google Gemini (API) as the conversation model, replacing the local Qwen SLM.

Same job as ptp_copilot.slm (which is kept unchanged): read the live dialogue prefix plus a
compact account context and return
  * the PTP category with a probability for each of the six types (structured JSON output),
  * up to three short reasons,
  * on request, a draft next agent line (display only; agents see the approved policy script).

Gemini is not fine-tuned: it is prompted with the six definitions, the labelling conventions
used to build our ground truth, and one short real training-split example per type.
Probabilities are temperature-calibrated on validation calls (outputs/gemini_calibration.json).

The API key is read from the GEMINI_API_KEY environment variable (on Windows also from the
user environment set with `setx`), never from code or git.
"""
import json
import os
import time

import numpy as np

from .config import CATEGORIES, OUT_DIR

MODEL = os.environ.get("PTP_GEMINI_MODEL", "gemini-3.5-flash-lite")
CALIB_FILE = OUT_DIR / "gemini_calibration.json"
SPK = {"agent": "A", "bot": "B", "customer": "C"}

DEFINITIONS = """You classify a live debt-collection call (India; Hinglish, Kanglish = Kannada-English, or English;
names and lenders are masked) into the promise-to-pay (PTP) type it most resembles, using only what has been said so far.

Types (apply in this priority order when several fit):
1. third_party: someone other than the borrower answered (spouse, parent, sibling, child, colleague: e.g. "main unki mummy hoon",
   "naanu avra amma", "this is his wife") AND they relay or give a date/timeframe for payment.
2. agent_recorded_or_pushed: the borrower never states a date or amount of their own; the agent proposes the date (often after
   pressure such as CIBIL damage, escalation, "last reminder") and the borrower only gives minimal acknowledgement ("haan", "ok",
   "theek hai"), or the agent records a PTP without any real commitment. A third party answering with NO commitment, where the
   agent still records a PTP, also belongs here.
3. genuine_infeasible: the borrower is sincere but cash flow does not fit: hardship such as salary late or on hold, job loss,
   medical emergency or hospital, business down, client not paid, no balance; often a conditional promise ("salary aayi toh"),
   a part-payment offer, or a date that does not fit the salary day. Severe hardship (medical, job loss) is always this type.
4. repeat_promiser: the account has a history of broken promises (prior_broken >= 2 in CTX) and the call refers to a missed
   earlier promise (agent cites it, borrower excuses it: "woh alag baat thi", "that was different") without a hardship-driven
   part-payment offer.
5. escape: evasive promise made to end the call: hedging ("dekhta hoon", "try karunga", "koshish karunga", "nodtini", "maybe",
   "let me see"), trying to hang up ("baad mein call karo", "meeting mein hoon", "I have to go"), vague dates ("agle hafte",
   "next week"), "forgot"; no firm date of their own.
6. genuine_feasible: the borrower clearly states their own date and/or amount, uses firm language ("pakka", "khud kar dunga",
   "for sure"), accepts the payment link, date aligned with salary; no strong negative signals.

Input format: a CTX line (language, portfolio, prior broken PTPs, salary day or ?, promise amount and date under discussion),
then CALL lines prefixed A: agent, B: voice bot, C: customer.
Return probabilities for all six types that sum to 1; be calibrated (do not put 1.0 unless certain)."""

DRAFT_RULES = """Write the agent's next line, in the borrower's language (Hinglish / Kanglish / English as in CTX), one or two sentences.
Rules: never threaten, shame or mention police/legal action; if a third party is on the line, never mention the loan, amount
or dues, only ask when the borrower is available; for hardship offer a date after salary, a smaller amount or restructuring;
for evasive promises ask for a small part payment now via payment link; never invent dates or amounts not in CTX/CALL."""


def api_key():
    k = os.environ.get("GEMINI_API_KEY")
    if not k and os.name == "nt":                       # `setx` only reaches new processes; read the user env directly
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                k = winreg.QueryValueEx(h, "GEMINI_API_KEY")[0]
        except OSError:
            k = None
    return (k or "").strip().strip('"') or None


def ctx_line(r):
    import math
    import pandas as pd
    sd = r.get("salary_credit_day")
    sd = "?" if sd is None or (isinstance(sd, float) and math.isnan(sd)) else int(sd)
    pdt = pd.Timestamp(r["promised_date"]).strftime("%d %b")
    return (f"CTX lang={r['preferred_language']} portfolio={r['portfolio']} prior_broken={int(r['total_prior_broken'])} "
            f"salary_day={sd} promise=Rs{int(r['promised_amount'])} on {pdt}")


def call_text(ctx_str, turns):
    from .lexicon import clean
    body = "\n".join(f"{SPK.get(s, 'A')}: {clean(x).strip()}" for s, x in turns)
    return f"{ctx_str}\nCALL:\n{body}"


def few_shot_examples():
    """One short, human-labelled training-split call per type (never validation/test)."""
    import pandas as pd
    df = pd.read_pickle(OUT_DIR / "slm_sft.pkl")
    gt = pd.read_csv(OUT_DIR / "ground_truth_labels.csv")[["ptp_id", "category_source"]]
    tr = df[(df["split"] == "train") & (df["is_final"] == 1)].merge(gt, on="ptp_id")
    tr = tr.assign(n=tr["turns"].map(lambda t: len(json.loads(t))))
    out = []
    for c in CATEGORIES:
        s = tr[(tr["category"] == c) & (tr["n"].between(5, 10))]
        s = s.assign(rank=s["category_source"].map({"human_annotation": 0, "pseudo_label_high_conf": 1}).fillna(2))
        s = s.sort_values(["rank", "n"])
        if len(s):
            r = s.iloc[0]
            out.append((call_text(r["ctx"].split("\n")[0], json.loads(r["turns"])), c))
    return out


def _schema():
    from google.genai import types
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "category": types.Schema(type=types.Type.STRING, enum=CATEGORIES),
            "probabilities": types.Schema(type=types.Type.OBJECT, properties={
                c: types.Schema(type=types.Type.NUMBER) for c in CATEGORIES}, required=CATEGORIES),
            "reasons": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), max_items=3),
        },
        required=["category", "probabilities", "reasons"],
    )


class GeminiClassifier:
    def __init__(self, model=MODEL):
        from google import genai
        key = api_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)
        self.model = model
        shots = "\n\n".join(f"Example ({c}):\n{t}" for t, c in few_shot_examples())
        self.system = DEFINITIONS + "\n\nLabelled examples from past calls:\n\n" + shots
        self.T = 1.0
        if CALIB_FILE.exists():
            self.T = float(json.loads(CALIB_FILE.read_text())["temperature"])

    def _config(self, system=None, **kw):
        from google.genai import types
        cfg = dict(system_instruction=system or self.system, temperature=0.0, **kw)
        try:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level="minimal")
            return types.GenerateContentConfig(**cfg)
        except Exception:
            cfg.pop("thinking_config", None)
            return types.GenerateContentConfig(**cfg)

    def classify_text(self, text, retries=5):
        from google.genai import types  # noqa: F401
        cfg = self._config(response_mime_type="application/json", response_schema=_schema())
        for i in range(retries):
            try:
                t0 = time.perf_counter()
                r = self.client.models.generate_content(model=self.model, contents=text, config=cfg)
                ms = (time.perf_counter() - t0) * 1000
                d = json.loads(r.text)
                p = np.array([max(float(d["probabilities"].get(c, 0.0)), 1e-6) for c in CATEGORIES])
                p = p / p.sum()
                z = np.log(p) / self.T
                pc = np.exp(z - z.max())
                pc = pc / pc.sum()
                usage = getattr(r, "usage_metadata", None)
                return dict(raw_category=d["category"], probs_raw=p, probs=pc, reasons=d.get("reasons", [])[:3],
                            latency_ms=ms, prompt_tokens=getattr(usage, "prompt_token_count", None),
                            output_tokens=getattr(usage, "candidates_token_count", None))
            except Exception as e:
                msg = str(e)
                if i == retries - 1:
                    raise
                time.sleep(min(30, 2 ** i * (4 if "429" in msg or "RESOURCE_EXHAUSTED" in msg else 1)))

    def draft(self, text):
        """Free-text next agent line (display only). Uses its own instructions: the classifier's
        system prompt asks for JSON probabilities, which is not what a draft line should be."""
        system = ("You are a debt-collection call copilot for Indian lenders. You read a live call "
                  "(CTX line, then A: agent / B: voice bot / C: customer lines) and suggest the agent's next line.\n"
                  + DRAFT_RULES + "\nReturn only the line itself: no labels, no JSON, no quotes.")
        t0 = time.perf_counter()
        r = self.client.models.generate_content(model=self.model, contents=text, config=self._config(system=system))
        return dict(text=(r.text or "").strip(), latency_ms=(time.perf_counter() - t0) * 1000)
