"""Phase 3b/4 - Small Language Model track: LoRA fine-tune + latency benchmark.

Model: Qwen2.5-0.5B-Instruct (multilingual tokenizer handles romanised Hinglish/Kanglish).
Task : live dialogue prefix (+ compact account context) ->
         "<code>|<category>|<next agent utterance>"
       The FIRST generated token is a single digit class code 1..6, so
         * the category and its confidence are available at time-to-first-token,
         * confidence = softmax over the six digit logits, temperature-calibrated,
         * the utterance streams afterwards and can be voiced/shown incrementally.
Targets: category = Phase-1 ground truth; utterance = the compliance-filtered
policy script (policy-distillation: the SLM learns to *say* only pre-approved,
FPC/DPDP-safe guidance, conditioned on the live conversation).

Run:  python -m ptp_copilot.slm train | eval | bench | all
"""
import json
import math
import os
import random
import sys
import time

# Optional: borrow a CUDA torch build from another local environment *read-only*
# (appended last, so this environment's own packages always win).
if os.environ.get("PTP_TORCH_SITE") and os.environ["PTP_TORCH_SITE"] not in sys.path:
    sys.path.append(os.environ["PTP_TORCH_SITE"])

import numpy as np
import pandas as pd

from . import features as F
from . import policy as P
from .config import CATEGORIES, OUT_DIR, SEED, TTFT_SLA_MS, VRAM_SLA_GB

BASE_DIR = OUT_DIR / "hf" / "Qwen2.5-0.5B-Instruct"
LORA_DIR = OUT_DIR / "slm_lora"
MERGED_DIR = OUT_DIR / "slm_merged"
HF_BASE_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
HF_MODEL_REPO = os.environ.get("PTP_HF_MODEL_REPO", "Shubhbagaria12/QWEN_OPENIIT")  # fine-tuned, merged weights
CODES = [str(i + 1) for i in range(len(CATEGORIES))]


def ensure_local(local_dir, repo_id):
    """Model weights are not in git (942 MB): download them from the Hugging Face Hub on first use."""
    if not (local_dir / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        print(f"[slm] {local_dir} missing; downloading {repo_id} from the Hugging Face Hub")
        snapshot_download(repo_id, local_dir=str(local_dir), allow_patterns=[
            "*.json", "*.safetensors", "*.jinja", "*.txt"], ignore_patterns=["lora_adapter/*"])
    return local_dir
SYS = ("You are a debt-collection call copilot. Read the live call and answer as "
       "<code>|<category>|<next agent line>. Codes: " +
       " ".join(f"{c}={n}" for c, n in zip(CODES, CATEGORIES)) +
       ". Never disclose debt details to a third party. Never threaten. Protect hardship cases.")
HEAD = "<|im_start|>system\n" + SYS + "<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"
SPK = {"agent": "A", "bot": "B", "customer": "C"}


def ctx_line(r):
    sd = r.get("salary_credit_day")
    sd = "?" if sd is None or (isinstance(sd, float) and math.isnan(sd)) else int(sd)
    pdt = pd.Timestamp(r["promised_date"]).strftime("%d %b")
    return (f"CTX lang={r['preferred_language']} portfolio={r['portfolio']} prior_broken={int(r['total_prior_broken'])} "
            f"salary_day={sd} promise=Rs{int(r['promised_amount'])} on {pdt}\nCALL:\n")


def turn_line(speaker, text):
    return f"{SPK.get(speaker, 'A')}: {F.lx.clean(text).strip()}\n"


# ============================================================================ data
def build_sft(max_prefix_per_call=3, min_frac=0.4):
    from .pipeline import build, add_conv
    D, lab, ts, pf, base, conv_cols = build()
    en = pd.read_csv(OUT_DIR / "enriched_ptp_predictions.csv")
    gt = en.set_index("ptp_id")
    tr = D["transcripts"]
    calls = base[base["attempt_id"].isin(tr["call_id"])].copy()
    pfx = pf.merge(ts[["call_id", "turn_idx", "speaker"]], on=["call_id", "turn_idx"])
    rng = random.Random(SEED)
    rows = []
    for r in calls.itertuples(index=False):
        r = r._asdict()
        t = tr[tr["call_id"] == r["attempt_id"]]
        n = len(t)
        p = pfx[pfx["call_id"] == r["attempt_id"]].reset_index(drop=True)
        cut_pts = [i for i in range(n) if t.iloc[i]["speaker"] == "customer" and (i + 1) / n >= min_frac and i < n - 1]
        if r["split"] == "train":
            cuts = rng.sample(cut_pts, min(len(cut_pts), max_prefix_per_call - 1)) + [n - 1]
        else:
            cuts = ([rng.choice(cut_pts)] if cut_pts else []) + [n - 1]
        g = gt.loc[r["ptp_id"]]
        cat = g["ground_truth_category"]
        for k in cuts:
            feats = p.iloc[k].to_dict()
            feats["hardship_30d"] = r["hardship_30d"]
            safe_cat, _ = P.apply_guardrails(cat, feats)
            utt = P.guidance(safe_cat, r["preferred_language"], feats, g["p_keep_in_call"], r["promised_amount"],
                             pd.Timestamp(r["promised_date"]).strftime("%d %b"), r["salary_credit_day"])
            body = "".join(turn_line(s, x) for s, x in zip(t["speaker"].iloc[:k + 1], t["text"].iloc[:k + 1]))
            rows.append(dict(ptp_id=r["ptp_id"], split=r["split"], is_final=int(k == n - 1), frac=(k + 1) / n,
                             prompt=HEAD + ctx_line(r) + body + TAIL,
                             target=f"{CODES[CATEGORIES.index(cat)]}|{cat}|{utt}",
                             category=cat, safe_category=safe_cat, utterance=utt,
                             turns=json.dumps([[s, x] for s, x in zip(t["speaker"], t["text"])], ensure_ascii=False),
                             ctx=ctx_line(r)))
    df = pd.DataFrame(rows)
    df.to_pickle(OUT_DIR / "slm_sft.pkl")
    print("[slm] SFT rows:", df.groupby(["split", "is_final"]).size().to_dict())
    return df


def _has_parquet():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def load_sft():
    p = OUT_DIR / "slm_sft.pkl"
    return pd.read_pickle(p) if p.exists() else pd.read_parquet(OUT_DIR / "slm_sft.parquet")


# ============================================================================ train
def train(epochs=2, lr=2e-4, bs=2, accum=8, max_len=900, max_train=2400, code_weight=5.0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(SEED)
    df = load_sft()
    trn = df[df["split"] == "train"].sample(frac=1, random_state=SEED).head(max_train)
    ensure_local(BASE_DIR, HF_BASE_REPO)
    tok = AutoTokenizer.from_pretrained(BASE_DIR)
    model = AutoModelForCausalLM.from_pretrained(BASE_DIR, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    def encode(p, t):
        a = tok(p, add_special_tokens=False)["input_ids"]
        b = tok(t + "<|im_end|>", add_special_tokens=False)["input_ids"]
        a = a[-(max_len - len(b)):]            # keep the most recent turns if too long
        return a + b, len(a)

    data = [encode(p, t) for p, t in zip(trn["prompt"], trn["target"])]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    steps = math.ceil(len(data) / (bs * accum)) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / steps))
    model.train()
    log, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        random.Random(SEED + ep).shuffle(data)
        for i in range(0, len(data), bs):
            batch = data[i:i + bs]
            L = max(len(x) for x, _ in batch)
            ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for j, (x, _) in enumerate(batch):
                ids[j, :len(x)] = torch.tensor(x)
                att[j, :len(x)] = 1
            ids, att = ids.cuda(), att.cuda()
            # memory: run the trunk, project ONLY completion positions through lm_head
            h = model.base_model.model.model(input_ids=ids, attention_mask=att).last_hidden_state
            losses = []
            for j, (x, a) in enumerate(batch):
                pos = torch.arange(a - 1, len(x) - 1, device=ids.device)
                logits = model.base_model.model.lm_head(h[j, pos]).float()
                ce = torch.nn.functional.cross_entropy(logits, ids[j, pos + 1], reduction="none")
                # the class-code token is 1 of ~45 target tokens; without its own weight the
                # model learns the (easy) templated utterance and collapses to the class prior
                losses.append((code_weight * ce[0] + ce[1:].mean()) / (code_weight + 1))
            loss = torch.stack(losses).mean() / accum
            loss.backward()
            if (i // bs + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    log.append(dict(step=step, loss=round(loss.item() * accum, 4), s=round(time.time() - t0, 1)))
                    print(f"[slm-train] step {step}/{steps} loss {loss.item() * accum:.4f} "
                          f"{time.time() - t0:.0f}s vram={torch.cuda.max_memory_allocated() / 1e9:.2f}GB", flush=True)
    model.save_pretrained(LORA_DIR)
    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_DIR)
    json.dump(dict(steps=step, train_rows=len(data), epochs=epochs, lr=lr, eff_batch=bs * accum, code_weight=code_weight,
                   minutes=round((time.time() - t0) / 60, 1), peak_train_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   loss_curve=log), open(OUT_DIR / "slm_train_log.json", "w"), indent=1)


# ============================================================================ load / score
def load(precision="fp16", path=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = path or ensure_local(MERGED_DIR, HF_MODEL_REPO)
    tok = AutoTokenizer.from_pretrained(path)
    if precision == "nf4":
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=q, device_map="cuda:0")
    elif precision == "int8":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                     device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).cuda()
    model.eval()
    code_ids = [tok(c, add_special_tokens=False)["input_ids"][0] for c in CODES]
    return tok, model, code_ids


def code_logits(tok, model, code_ids, prompts, bs=8):
    import torch
    out = []
    tok.padding_side = "left"
    for i in range(0, len(prompts), bs):
        enc = tok(list(prompts[i:i + bs]), return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[:, -1, :].float()
        out.append(lg[:, code_ids].cpu().numpy())
    return np.concatenate(out)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ============================================================================ evaluate
def evaluate(n_gen=150):
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from .labels import _fit_temperature
    from .models import multiclass_ece
    df = load_sft()
    tok, model, code_ids = load("fp16")
    va = df[(df["split"] == "validation") & (df["is_final"] == 1)]
    te = df[df["split"] == "test"]
    yv = va["category"].map(CATEGORIES.index).to_numpy()
    lv = code_logits(tok, model, code_ids, va["prompt"].tolist())
    T = _fit_temperature(np.log(softmax(lv)), yv)
    lt = code_logits(tok, model, code_ids, te["prompt"].tolist())
    Pt = softmax(lt, T)
    yt = te["category"].map(CATEGORIES.index).to_numpy()
    res = {"temperature": round(T, 3)}
    for name, m in [("final", te["is_final"].to_numpy() == 1), ("mid_call_prefix", te["is_final"].to_numpy() == 0)]:
        pr = Pt[m].argmax(1)
        res[name] = dict(n=int(m.sum()), acc=round(accuracy_score(yt[m], pr), 4),
                         macro_f1=round(f1_score(yt[m], pr, average="macro"), 4),
                         ece_top1_calibrated=round(multiclass_ece(yt[m], Pt[m]), 4),
                         ece_top1_uncalibrated=round(multiclass_ece(yt[m], softmax(lt[m])), 4))
    m = te["is_final"].to_numpy() == 1
    pr = Pt[m].argmax(1)
    labels = list(range(len(CATEGORIES)))
    res["final_report"] = classification_report(yt[m], pr, labels=labels, target_names=CATEGORIES, digits=3,
                                                zero_division=0)
    res["final_confusion"] = confusion_matrix(yt[m], pr, labels=labels).tolist()
    gi, agg = CATEGORIES.index("genuine_infeasible"), [CATEGORIES.index("escape"), CATEGORIES.index("repeat_promiser")]
    res["hardship_to_aggressive_rate"] = round(float(np.isin(pr[yt[m] == gi], agg).mean()), 4)
    print(res["final_report"])

    # generation quality on a sample: exact match to policy script + compliance
    sm = te[te["is_final"] == 1].sample(min(n_gen, int(m.sum())), random_state=SEED)
    tok.padding_side = "left"
    exact, fmt_ok, comp_bad, comp_bad_served, same_cat = 0, 0, 0, 0, 0
    samples = []
    for i in range(0, len(sm), 8):
        chunk = sm.iloc[i:i + 8]
        enc = tok(chunk["prompt"].tolist(), return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=110, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (_, r), o in zip(chunk.iterrows(), txt):
            parts = o.strip().split("|", 2)
            ok = len(parts) == 3 and parts[0].strip() in CODES
            fmt_ok += ok
            utt = parts[2].strip() if ok else o
            cat = CATEGORIES[int(parts[0]) - 1] if ok else "genuine_feasible"
            raw_bad = bool(P.compliance_check(utt, r["safe_category"]))
            comp_bad += raw_bad
            # serving path: deterministic lexical guardrail over the same transcript, then the filter
            served_cat, served_utt = serve_guard(cat, utt, r)
            comp_bad_served += bool(P.compliance_check(served_utt, r["safe_category"]))
            exact += utt == r["utterance"]
            same_cat += cat == r["category"]
            if len(samples) < 12:
                samples.append(dict(ptp_id=r["ptp_id"], target=r["target"], generated=o.strip()))
    res["generation"] = dict(n=len(sm), format_valid=round(fmt_ok / len(sm), 4),
                             utterance_exact_match=round(exact / len(sm), 4),
                             compliance_violations_raw_slm=int(comp_bad),
                             compliance_violations_after_serving_guardrail=int(comp_bad_served),
                             category_match=round(same_cat / len(sm), 4))
    res["generation_samples"] = samples
    print("[slm-eval]", json.dumps({k: v for k, v in res.items() if k not in ("final_report", "generation_samples")},
                                   indent=1))
    json.dump(res, open(OUT_DIR / "slm_eval.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return res


def serve_guard(cat, utt, r):
    """Production wrapper: lexical third-party / hardship detectors (no model) can only
    soften the SLM's choice; the utterance is then re-rendered from the approved script."""
    turns = json.loads(r["turns"])
    cust = " ".join(F.lx.clean(x) for s, x in turns if s == "customer")
    feats = {"n_c_third_party": int(bool(F.lx.THIRD_PARTY.search(cust) or F.lx.RELAY.search(cust))),
             "n_c_wrong_number": int(bool(F.lx.WRONG_NUMBER.search(cust))),
             "n_c_hardship": int(bool(F.lx.HARDSHIP.search(cust))),
             "n_c_severe_hardship": int(bool(F.lx.SEVERE_HARDSHIP.search(cust)))}
    safe, rails = P.apply_guardrails(cat, feats)
    if safe == cat:
        return cat, utt
    import re as _re
    lang = r["ctx"].split("lang=")[1].split()[0]
    m = _re.search(r"promise=Rs(\d+) on (\d+ \w+)", r["ctx"])
    sd = _re.search(r"salary_day=(\d+)", r["ctx"])
    amt, date = (float(m.group(1)), m.group(2)) if m else (0.0, "")
    return safe, P.guidance(safe, lang, {}, 0.5, amt, date, float(sd.group(1)) if sd else None)


# ============================================================================ latency benchmark
def benchmark(n_calls=40, new_tokens=48, precisions=("fp16", "nf4")):
    """CUDA-synchronised latency: TTFT (prefill+first token), decode tok/s, E2E, peak VRAM.

    Two serving modes:
      cold        - whole prefix re-encoded every turn (naive)
      incremental - KV cache of the call so far is kept; each new utterance only
                    prefills its own tokens (+ the 5-token assistant header), then
                    the header is cropped back off the cache."""
    import torch
    from transformers import DynamicCache
    df = load_sft()
    te = df[(df["split"] == "test") & (df["is_final"] == 1)].sample(n_calls, random_state=SEED)
    rows = []
    for prec in precisions:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        tok, model, code_ids = load(prec)
        weights_gb = torch.cuda.memory_allocated() / 1e9
        tail_ids = tok(TAIL, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        # warm-up
        for p in te["prompt"].head(3):
            ids = tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            with torch.no_grad():
                model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        cold_ttft, cold_e2e, tps, inc_ttft, prompt_len, agree = [], [], [], [], [], []
        for r in te.itertuples(index=False):
            ids = tok(r.prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            prompt_len.append(ids.shape[1])
            # ---- cold TTFT: full prefill + first token
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(ids, use_cache=True)
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1000
            cold_ttft.append(ttft)
            # ---- decode throughput (greedy, manual loop with cache)
            past = out.past_key_values
            t1 = time.perf_counter()
            n_dec = 0
            with torch.no_grad():
                for _ in range(new_tokens - 1):
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1].argmax(-1, keepdim=True)
                    n_dec += 1
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            tps.append(n_dec / dt)
            cold_e2e.append(ttft + dt * 1000)
            # ---- incremental per-turn TTFT over the whole call
            turns = json.loads(r.turns)
            head_ids = tok(HEAD + r.ctx, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            cache = DynamicCache()
            with torch.no_grad():
                model(head_ids, past_key_values=cache, use_cache=True)
            for s, x in turns:
                t_ids = tok(turn_line(s, x), add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    o = model(torch.cat([t_ids, tail_ids], 1), past_key_values=cache, use_cache=True)
                    code = o.logits[:, -1, code_ids].argmax(-1)
                torch.cuda.synchronize()
                inc_ttft.append((time.perf_counter() - t0) * 1000)
                cache.crop(-tail_ids.shape[1])     # drop the assistant header, keep the new turn
            agree.append(int(code.item()) == int(out.logits[0, -1, code_ids].argmax().item()))
        peak = torch.cuda.max_memory_allocated() / 1e9
        q = lambda a, p: round(float(np.percentile(a, p)), 1)
        rows.append(dict(precision=prec, weights_gb=round(weights_gb, 3), peak_vram_gb=round(peak, 3),
                         vram_sla_pass=peak < VRAM_SLA_GB, mean_prompt_tokens=int(np.mean(prompt_len)),
                         cold_ttft_p50_ms=q(cold_ttft, 50), cold_ttft_p95_ms=q(cold_ttft, 95),
                         incremental_ttft_p50_ms=q(inc_ttft, 50), incremental_ttft_p95_ms=q(inc_ttft, 95),
                         ttft_sla_pass=q(inc_ttft, 95) < TTFT_SLA_MS,
                         decode_tok_per_s=round(float(np.mean(tps)), 1),
                         e2e_ms_48_tokens_p50=q(cold_e2e, 50), incremental_vs_cold_code_agreement=round(np.mean(agree), 3)))
        print("[bench]", rows[-1], flush=True)
        del model
    res = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=rows)
    json.dump(res, open(OUT_DIR / "latency_slm.json", "w"), indent=1, default=float)
    return res


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("data", "all"):
        build_sft()
    if cmd in ("train", "all"):
        train()
    if cmd in ("eval", "all"):
        evaluate()
    if cmd in ("bench", "all"):
        benchmark()
