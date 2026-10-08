"""Replay held-out calls through the live copilot; benchmark per-turn CPU latency.

Run:  python -m ptp_copilot.demo_streaming
"""
import json

import numpy as np
import pandas as pd

from .config import OUT_DIR
from .pipeline import build
from .streaming import InCallCopilot, replay


def main(n_bench=120):
    D, lab, ts, pf, base, conv_cols = build()
    en = pd.read_csv(OUT_DIR / "enriched_ptp_predictions.csv")
    cop = InCallCopilot()
    tr = D["transcripts"]
    test = base[(base["split"] == "test") & base["attempt_id"].isin(tr["call_id"])]
    test = test.merge(en[["ptp_id", "ground_truth_category"]], on="ptp_id")

    # ---- latency benchmark over many calls
    lat, n_turns = [], 0
    for _, r in test.sample(min(n_bench, len(test)), random_state=0).iterrows():
        out = replay(cop, r.to_dict(), tr[tr["call_id"] == r["attempt_id"]])
        lat += [o["latency_ms"] for o in out[1:]]          # skip first-call warm-up per call
        n_turns += len(out)
    lat = np.array(lat)
    bench = dict(calls=min(n_bench, len(test)), turns=int(n_turns), p50_ms=round(np.percentile(lat, 50), 2),
                 p95_ms=round(np.percentile(lat, 95), 2), p99_ms=round(np.percentile(lat, 99), 2),
                 max_ms=round(lat.max(), 2), device="CPU (feature extraction + LightGBM + policy)")
    print("[latency] tabular copilot per-turn update:", bench)

    # ---- illustrative turn-by-turn transcripts, one per category
    md = ["# Live copilot replays (held-out test calls)\n",
          "Each row is the copilot's state immediately after that utterance. `heard_enough` marks the first point "
          "where the score is stable and a commitment event has occurred.\n"]
    for cat in ["escape", "agent_recorded_or_pushed", "third_party", "genuine_infeasible", "repeat_promiser",
                "genuine_feasible"]:
        cand = test[test["ground_truth_category"] == cat]
        if cand.empty:
            continue
        r = cand.iloc[0]
        out = replay(cop, r.to_dict(), tr[tr["call_id"] == r["attempt_id"]])
        md.append(f"\n## {r['ptp_id']} - ground truth `{cat}` - outcome: {'KEPT' if r['kept'] else 'NOT KEPT'} "
                  f"({r['status']}), lang={r['preferred_language']}\n")
        md.append("| # | speaker | utterance | p_keep | category (conf) | heard enough | copilot suggestion |")
        md.append("|---|---|---|---|---|---|---|")
        fired = False
        for o in out:
            he = "**YES**" if o["heard_enough"] and not fired else ""
            fired = fired or o["heard_enough"]
            rails = f" ⚑{','.join(o['guardrails'])}" if o["guardrails"] else ""
            sug = o["next_utterance"] if o["speaker"] == "customer" else ""
            md.append(f"| {o['turn']} | {o['speaker']} | {str(o['text']).replace('|', '/')} | {o['p_keep']:.2f} | "
                      f"{o['category']} ({o['category_conf']:.2f}){rails} | {he} | {sug} |")
    (OUT_DIR / "streaming_demo.md").write_text("\n".join(md), encoding="utf-8")
    with open(OUT_DIR / "latency_tabular.json", "w") as fh:
        json.dump(bench, fh, indent=1, default=float)
    return bench


if __name__ == "__main__":
    main()
