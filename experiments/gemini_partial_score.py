"""Early score: Gemini on the test calls completed so far, with Qwen and LightGBM scored on
exactly the same calls for a like-for-like comparison. Writes outputs/gemini_eval.json.

Run from the repository root:  python experiments/gemini_partial_score.py
(Qwen needs CUDA torch; set PTP_TORCH_SITE if torch lives in another environment.)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptp_copilot.config import CATEGORIES, OUT_DIR       # noqa: E402
from ptp_copilot.models import multiclass_ece            # noqa: E402

sys.path.insert(0, str(ROOT / "experiments"))
from gemini_eval import CACHE, metrics                   # noqa: E402

T = json.loads((OUT_DIR / "gemini_calibration.json").read_text())["temperature"]
recs = [json.loads(l) for l in CACHE.read_text(encoding="utf-8").splitlines()]
test = [r for r in recs if r["key"].startswith("test_final|")]
val = [r for r in recs if r["key"].startswith("val|")]
keys = [r["key"].split("|", 1)[1] for r in test]

df = pd.read_pickle(OUT_DIR / "slm_sft.pkl").reset_index().rename(columns={"index": "key"})
df["key"] = df["key"].astype(str)
sub = df.set_index("key").loc[keys].reset_index()
y = sub["category"].map(CATEGORIES.index).to_numpy()
gi, agg = CATEGORIES.index("genuine_infeasible"), [CATEGORIES.index("escape"), CATEGORIES.index("repeat_promiser")]


def summary(pred, P=None):
    d = dict(n=len(pred), acc=round(accuracy_score(y, pred), 4), macro_f1=round(f1_score(y, pred, average="macro"), 4),
             hardship_to_aggressive=round(float(np.isin(pred[y == gi], agg).mean()), 4) if (y == gi).any() else None)
    if P is not None:
        d["ece"] = round(multiclass_ece(y, P), 4)
    return d


res = dict(status=f"PARTIAL: {len(test)} of 484 held-out test calls (run stopped early to save quota)",
           model="gemini-3.5-flash-lite", temperature=round(T, 3), n_validation=len(val))
res["final"] = metrics(test, T)

# ---- LightGBM (deployed, cost-sensitive) on the same calls: stored out-of-fold decisions
en = pd.read_csv(OUT_DIR / "enriched_ptp_predictions.csv").set_index("ptp_id")
lgb_pred = sub["ptp_id"].map(en["category_model"]).map(CATEGORIES.index).to_numpy()
lgb_P = sub["ptp_id"].map(lambda p: [en.loc[p, "p_" + c] for c in CATEGORIES]).apply(np.array)
res["same_calls_lightgbm"] = summary(lgb_pred, np.stack(lgb_P.to_numpy()))

# ---- Qwen fp16 on the same calls (re-scored locally, GPU)
try:
    from ptp_copilot import slm
    q = json.load(open(OUT_DIR / "slm_eval.json", encoding="utf-8"))
    tok, model, code_ids = slm.load("fp16")
    lg = slm.code_logits(tok, model, code_ids, sub["prompt"].tolist())
    Pq = slm.softmax(lg, q["temperature"])
    res["same_calls_qwen_fp16"] = summary(Pq.argmax(1), Pq)
except Exception as e:
    res["same_calls_qwen_fp16"] = f"not scored: {type(e).__name__}: {e}"

g = res["final"]
res["same_calls_gemini"] = dict(n=g["n"], acc=g["acc"], macro_f1=g["macro_f1"],
                                hardship_to_aggressive=g["hardship_to_aggressive_rate"], ece=g["ece_top1_calibrated"])
res["full_test_reference"] = dict(qwen_fp16_484=dict(acc=0.6446, macro_f1=0.6731), lightgbm_563=dict(acc=0.7016, macro_f1=0.7479))
json.dump(res, open(OUT_DIR / "gemini_eval.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "final"}, indent=1))
print("gemini per-class F1:", g["per_class_f1"])
print("gemini latency (ms):", g["latency_ms"], "| tokens in/out:", g["mean_prompt_tokens"], g["mean_output_tokens"])
print("gemini class support in subset:", np.bincount(y, minlength=6).tolist(), CATEGORIES)
