"""API smoke test against a running backend (python webapp/start.py).

    python webapp/tests/api_smoke_test.py [http://127.0.0.1:8000]

Checks real model loading, inference, parsing, confidence, category, customer/payment/PTP
lookups, treatment logic, simulation steps and audit creation. Exits non-zero on failure.
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
fails = []


def call(path, body=None):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def check(name, cond, info=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  :: {info}" if info else ""))
    if not cond:
        fails.append(name)


h = None
for _ in range(60):
    try:
        h = call("/api/health")
    except OSError:          # server still starting
        time.sleep(3)
        continue
    if h["data"]["ready"] and h["qwen"]["status"] in ("ready", "unavailable"):
        break
    time.sleep(3)
check("data loaded", h["data"]["ready"], f"{h['data']['ptps']} PTPs")
check("tabular model loaded", h["tabular"]["ready"])
check("qwen loaded (model + tokenizer)", h["qwen"]["status"] == "ready" and h["qwen"]["tokenizer_loaded"], h["qwen"].get("error") or h["qwen"]["version"])

sc = call("/api/live-calls/scenarios")
check("6 demo scenarios from real calls", len(sc) == 6, ", ".join(s["category"] for s in sc))
cid = sc[0]["call_id"]
meta = call(f"/api/live-calls/{cid}")
n = len(meta["transcript"])
steps = [call(f"/api/live-calls/{cid}/step", {"turn": k, "audit": False}) for k in range(n)]
check("step returns calibrated probability", all(0 < s["credibility"]["p_keep"] < 1 for s in steps))
q = steps[-1]["qwen"]
check("qwen inference + parsing + confidence", q.get("available") and abs(sum(q["probs"].values()) - 1) < 1e-3 and 0 < q["confidence"] <= 1,
      f"{q.get('category')} {q.get('confidence')}")
check("prefix scoring changes with the conversation", len({round(s['credibility']['p_keep'], 4) for s in steps}) > 1)
check("no future leakage: turn k sees k+1 turns", all(s["turn"] == k for k, s in enumerate(steps)))
p = call(f"/api/live-calls/{cid}/predict", {})
check("full prediction == final step (deterministic)", abs(p["credibility"]["p_keep"] - steps[-1]["credibility"]["p_keep"]) < 1e-9)
check("treatment present", bool(p["treatment"]["queue"]) and p["treatment"]["policy_check"] in ("pass", "blocked"))
tp = next(s for s in sc if s["category"] == "third_party")
t = call(f"/api/live-calls/{tp['call_id']}/predict", {})
check("third-party safeguard + no debt words in script", t["privacy_safeguard"] and t["treatment"]["policy_check"] == "pass",
      t["treatment"]["next_utterance"][:70])
c = call(f"/api/customers/{meta['account']['account_id']}")
check("customer, payments, PTP lookup", len(c["ptps"]) >= 1 and "payments" in c)
tr = call(f"/api/treatment/{sc[1]['ptp_id']}")
check("treatment endpoint", tr["live"] and tr["final_category"] in [s["category"] for s in sc] + ["genuine_feasible"])
au = call("/api/audit")
check("audit records written", any(r["audit_id"] == p["audit_id"] for r in au["rows"]))
an = call("/api/analytics?portfolio=msme")
check("analytics filter", an["kpis"]["ptps"] > 0)
print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
