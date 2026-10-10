"""Score Gemini on exactly the held-out calls used for the Qwen SLM evaluation.

  1. calibration: 200 validation calls (end of call) -> temperature T  -> outputs/gemini_calibration.json
  2. test: all 484 test calls at end of call + 484 mid-call prefixes (same rows as slm_eval.json)
Raw responses are cached in outputs/gemini_eval_raw.jsonl so the run can resume after network errors.
Results -> outputs/gemini_eval.json

Run from the repository root:  python experiments/gemini_eval.py
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ptp_copilot.config import CATEGORIES, OUT_DIR                  # noqa: E402
from ptp_copilot.gemini_llm import CALIB_FILE, GeminiClassifier, call_text  # noqa: E402
from ptp_copilot.labels import _fit_temperature                     # noqa: E402
from ptp_copilot.models import multiclass_ece                       # noqa: E402

CACHE = OUT_DIR / "gemini_eval_raw.jsonl"
WORKERS = 1
MIN_INTERVAL_S = 4.5          # ~13 requests/minute: measured sustainable on the free tier
N_VAL = 100
RUN_MID = "--mid" in sys.argv
lock = threading.Lock()
pace_lock = threading.Lock()
_last = [0.0]


def pace():
    with pace_lock:
        wait = _last[0] + MIN_INTERVAL_S - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def prefix_turns(r):
    """Turns visible at this prompt's cut point (the SFT prompt holds the prefix)."""
    turns = json.loads(r["turns"])
    n_lines = r["prompt"].split("CALL:\n", 1)[1].split("<|im_end|>")[0].strip().count("\n") + 1
    return turns[:n_lines]


def run(rows, clf, tag):
    done = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["key"]] = d
    todo = [r for r in rows if f"{tag}|{r['key']}" not in done]
    print(f"[{tag}] {len(rows)} rows, {len(todo)} to call", flush=True)

    def work(r):
        text = call_text(r["ctx"].split("\n")[0], prefix_turns(r))
        pace()
        out = clf.classify_text(text)
        rec = dict(key=f"{tag}|{r['key']}", category=r["category"], raw_category=out["raw_category"],
                   probs_raw=out["probs_raw"].tolist(), latency_ms=out["latency_ms"],
                   prompt_tokens=out["prompt_tokens"], output_tokens=out["output_tokens"])
        with lock, open(CACHE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    errors = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
                done[rec["key"]] = rec
            except Exception as e:
                errors += 1
                print("  error:", str(e)[:160], flush=True)
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    return [done[f"{tag}|{r['key']}"] for r in rows if f"{tag}|{r['key']}" in done], errors


def metrics(recs, T):
    y = np.array([CATEGORIES.index(r["category"]) for r in recs])
    P = np.array([r["probs_raw"] for r in recs])
    z = np.log(np.clip(P, 1e-9, 1)) / T
    Pc = np.exp(z - z.max(1, keepdims=True))
    Pc /= Pc.sum(1, keepdims=True)
    pred = Pc.argmax(1)
    gi, agg = CATEGORIES.index("genuine_infeasible"), [CATEGORIES.index("escape"), CATEGORIES.index("repeat_promiser")]
    lat = [r["latency_ms"] for r in recs]
    rep = classification_report(y, pred, labels=list(range(6)), target_names=CATEGORIES, digits=3, zero_division=0,
                                output_dict=True)
    return dict(n=len(recs), acc=round(accuracy_score(y, pred), 4),
                macro_f1=round(f1_score(y, pred, average="macro"), 4),
                weighted_f1=round(f1_score(y, pred, average="weighted"), 4),
                ece_top1_calibrated=round(multiclass_ece(y, Pc), 4),
                ece_top1_uncalibrated=round(multiclass_ece(y, P / P.sum(1, keepdims=True)), 4),
                hardship_to_aggressive_rate=round(float(np.isin(pred[y == gi], agg).mean()), 4),
                per_class_f1={c: round(rep[c]["f1-score"], 3) for c in CATEGORIES},
                confusion=confusion_matrix(y, pred, labels=list(range(6))).tolist(),
                latency_ms=dict(p50=round(float(np.percentile(lat, 50))), p95=round(float(np.percentile(lat, 95))),
                                mean=round(float(np.mean(lat)))),
                mean_prompt_tokens=round(float(np.mean([r["prompt_tokens"] or 0 for r in recs]))),
                mean_output_tokens=round(float(np.mean([r["output_tokens"] or 0 for r in recs]))))


def main():
    T0 = time.time()
    df = pd.read_pickle(OUT_DIR / "slm_sft.pkl").reset_index().rename(columns={"index": "key"})
    df["key"] = df["key"].astype(str)
    cached = set()
    if CACHE.exists():
        cached = {json.loads(l)["key"].split("|", 1)[1] for l in CACHE.read_text(encoding="utf-8").splitlines()
                  if json.loads(l)["key"].startswith("val|")}
    pool = df[(df.split == "validation") & (df.is_final == 1)].sample(200, random_state=0)
    pool = pd.concat([pool[pool["key"].isin(cached)], pool[~pool["key"].isin(cached)]])   # reuse paid-for calls first
    val = pool.head(N_VAL).to_dict("records")
    te_final = df[(df.split == "test") & (df.is_final == 1)].to_dict("records")
    te_mid = df[(df.split == "test") & (df.is_final == 0)].to_dict("records")
    clf = GeminiClassifier()
    clf.T = 1.0                                                     # raw probabilities; calibrate below
    v, ev = run(val, clf, "val")
    yv = np.array([CATEGORIES.index(r["category"]) for r in v])
    T = _fit_temperature(np.log(np.clip(np.array([r["probs_raw"] for r in v]), 1e-9, 1)), yv)
    CALIB_FILE.write_text(json.dumps(dict(temperature=T, model=clf.model, n_validation=len(v))))
    f, ef = run(te_final, clf, "test_final")
    m, em = run(te_mid, clf, "test_mid") if RUN_MID else ([], 0)
    res = dict(model=clf.model, temperature=round(T, 3), errors=dict(val=ev, final=ef, mid=em),
               validation=metrics(v, T), final=metrics(f, T),
               mid_call_prefix=metrics(m, T) if m else "not run (free-tier quota); use --mid",
               runtime_min=round((time.time() - T0) / 60, 1))
    # comparison with the stored results of the other models (same test rows)
    q = json.load(open(OUT_DIR / "slm_eval.json", encoding="utf-8"))
    lat = json.load(open(OUT_DIR / "latency_slm.json"))
    R = json.load(open(OUT_DIR / "results.json"))
    res["comparison_test_final"] = {
        "gemini": dict(acc=res["final"]["acc"], macro_f1=res["final"]["macro_f1"], ece=res["final"]["ece_top1_calibrated"],
                       hardship_to_aggressive=res["final"]["hardship_to_aggressive_rate"]),
        "qwen_fp16": dict(acc=q["final"]["acc"], macro_f1=q["final"]["macro_f1"], ece=q["final"]["ece_top1_calibrated"],
                          hardship_to_aggressive=q["hardship_to_aggressive_rate"]),
        "qwen_nf4": lat.get("nf4_quality_test_final"),
        "lightgbm_cost_sensitive_563_ptps": dict(
            acc=round(R["category_vs_ground_truth"]["report"]["accuracy"], 4),
            macro_f1=round(R["category_vs_ground_truth"]["report"]["macro avg"]["f1-score"], 4),
            ece=R["category_ece_top1"], hardship_to_aggressive=R["hardship_misclassified_as_aggressive_costsens"]),
    }
    json.dump(res, open(OUT_DIR / "gemini_eval.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("validation",)}, indent=1))


if __name__ == "__main__":
    main()
