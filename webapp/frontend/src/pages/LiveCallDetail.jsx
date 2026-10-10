import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { liveCallsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { llmLabel, llmReady, useHealth } from "../components/Health";
import { Async, CAT_LABEL, Card, CategoryPill, QUEUE_LABEL, Radial, StatusBadge, STATUS_LABEL, Unavailable, fmtInr, fmtPct } from "../components/ui";

const SIG_TAG = {
  c_date: ["date stated", "pos"], c_amount: ["amount stated", "pos"], c_partial_offer: ["part payment", "pos"], c_firm: ["firm", "pos"],
  c_link_accept: ["link accepted", "pos"], c_hedge: ["hedging", "neg"], c_exit: ["exit attempt", "neg"], c_forgot: ["forgot", "neg"],
  c_vague_date: ["vague date", "neg"], c_conditional: ["conditional", "neg"], c_link_refuse: ["link refused", "neg"], c_refusal: ["refusal", "neg"],
  c_prior_break_excuse: ["past-break excuse", "neg"], c_ack_to_proposal: ["passive ack", "neg"], a_pressure: ["pressure", "neg"],
  a_propose_date: ["agent proposed date", ""], a_prior_break: ["cited past break", ""], c_hardship: ["hardship", "alert"],
  c_severe_hardship: ["severe hardship", "alert"], c_third_party: ["third party", "alert"], c_wrong_number: ["wrong number", "alert"],
};
const OWN = {
  none_yet: ["No commitment yet", "neu"], borrower_originated: ["Borrower-originated", "pos"],
  agent_proposed_borrower_confirmed: ["Agent-proposed, borrower confirmed", "neu"], agent_proposed: ["Agent-proposed (weak)", "neg"],
};

export default function LiveCallDetail() {
  const { callId } = useParams();
  const nav = useNavigate();
  const { health } = useHealth();
  const meta = useApi((o) => liveCallsApi.get(callId, o), [callId]);
  const scenarios = useApi((o) => liveCallsApi.scenarios(o));
  const [status, setStatus] = useState("idle"); // idle | running | paused | stopped | done
  const [pos, setPos] = useState(0); // number of turns revealed
  const [assess, setAssess] = useState({}); // turn -> assessment
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [speed, setSpeed] = useState(2);
  const [useLlm, setUseLlm] = useState(true);
  const [llmRes, setLlmRes] = useState({}); // turn -> async LLM result
  const [llmWorking, setLlmWorking] = useState(false);
  const llmBusy = useRef(false);
  const llmPending = useRef(null);
  const llmFn = useRef(null);
  const [gen, setGen] = useState(null);
  const timer = useRef(null);
  const ctrl = useRef(null);
  const statusRef = useRef(status);
  statusRef.current = status;
  const tx = meta.data?.transcript || [];
  const n = tx.length;
  const llmOk = llmReady(health);
  const LLM = llmLabel(health);

  const reset = useCallback(() => {
    clearTimeout(timer.current);
    ctrl.current?.abort();
    setStatus("idle"); setPos(0); setAssess({}); setErr(null); setGen(null); setBusy(false);
    setLlmRes({}); llmPending.current = null;
  }, []);
  useEffect(() => reset, [callId, reset]);

  llmFn.current = async (turn) => {
    if (llmBusy.current) { llmPending.current = turn; return; }
    llmBusy.current = true; setLlmWorking(true);
    try { const r = await liveCallsApi.llm(callId, { turn }); setLlmRes((m) => ({ ...m, [turn]: r })); }
    catch { /* shown as unavailable for that turn */ }
    finally {
      llmBusy.current = false; setLlmWorking(false);
      const p = llmPending.current; llmPending.current = null;
      if (p !== null && p > turn) llmFn.current(p);
    }
  };

  const score = useCallback(async (turn) => {
    ctrl.current = new AbortController();
    setBusy(true);
    try {
      const last = turn === n - 1;   // final turn: include the LLM in the (audited) assessment
      const a = await liveCallsApi.step(callId, { turn, llm: last && useLlm && llmOk }, { signal: ctrl.current.signal });
      setAssess((m) => ({ ...m, [turn]: a }));
      if (last && a.llm?.available) setLlmRes((m) => ({ ...m, [turn]: { ...a.llm, turn } }));
      else if (useLlm && llmOk) llmFn.current(turn);
      setErr(null);
      return a;
    } catch (e) {
      if (e.name !== "AbortError") { setErr(e); setStatus("paused"); }
      return null;
    } finally {
      setBusy(false);
    }
  }, [callId, useLlm, llmOk, n]);

  // simulation loop: reveal next turn after its real recorded gap / speed, then score the prefix
  // The next utterance is revealed only after the previous one has been scored, so the replay
  // adapts to the server's speed (e.g. a free cloud instance) instead of queueing requests.
  useEffect(() => {
    if (status !== "running") return;
    if (busy) return;
    if (pos >= n) { setStatus("done"); return; }
    const prev = tx[pos - 1];
    const cur = tx[pos];
    const gap = prev ? Math.max(0, cur.start_ms - prev.start_ms) : 400;
    const delay = Math.min(4000, Math.max(250, gap / speed));
    timer.current = setTimeout(async () => {
      if (statusRef.current !== "running") return;
      setPos(pos + 1);
      await score(pos);
    }, pos === 0 ? 300 : delay);
    return () => clearTimeout(timer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status, pos, n, speed, busy]);

  const lastTurn = Math.max(-1, ...Object.keys(assess).map(Number));
  const a = lastTurn >= 0 ? assess[lastTurn] : null;
  const timeline = Object.values(assess).sort((x, y) => x.turn - y.turn).map((x) => ({
    turn: x.turn + 1, credibility: x.credibility.p_keep, llm_conf: llmRes[x.turn]?.available ? llmRes[x.turn].confidence : null,
    tab_conf: x.tabular_category.confidence,
  }));

  // "heard enough": first turn where the score is stable and a commitment / third-party event exists
  const ordered = Object.values(assess).sort((x, y) => x.turn - y.turn);
  let heardTurn = null;
  for (let i = 1; i < ordered.length && heardTurn === null; i++) {
    const p = ordered[i - 1], c = ordered[i];
    const stable = Math.abs(p.credibility.p_keep - c.credibility.p_keep) < 0.05 && p.final_category === c.final_category &&
      c.tabular_category.confidence >= 0.5;
    const event = c.commitment_event;   // commitment / date proposal / PTP recorded / third party (backend)
    if (stable && event) heardTurn = c.turn;
  }

  // latest LLM opinion available at or before the current turn
  const llmTurns = Object.keys(llmRes).map(Number).filter((t) => a && t <= a.turn && llmRes[t]?.available);
  const L = llmTurns.length ? llmRes[Math.max(...llmTurns)] : (a?.llm?.available ? { ...a.llm, turn: a.turn } : null);

  const runFull = async () => {
    reset();
    setBusy(true);
    try {
      const r = await liveCallsApi.predict(callId);
      setPos(n); setAssess({ [r.turn]: r }); setStatus("done");
      if (r.llm?.available) setLlmRes({ [r.turn]: { ...r.llm, turn: r.turn } });
    } catch (e) { setErr(e); } finally { setBusy(false); }
  };
  const runGen = async () => {
    setGen({ loading: true });
    try { setGen(await liveCallsApi.generate(callId, { turn: Math.max(0, pos - 1) })); }
    catch (e) { setGen({ error: e }); }
  };
  const tagsFor = (i) => (a?.evidence || []).filter((e) => e.first_turn === i && SIG_TAG[e.signal]);

  return (
    <Async q={meta} rows={10}>
      {(m) => (
        <>
          <div className="page-head">
            <div>
              <h1>Live call {m.call.call_id}</h1>
              <p className="row" style={{ gap: 8 }}>
                <StatusBadge s="LIVE DEMO SIMULATION" />
                <span className="small">{m.mode}</span>
              </p>
            </div>
            <div className="row">
              <select className="select" aria-label="Demo scenario" value={scenarios.data?.find((s) => s.call_id === callId) ? callId : ""}
                onChange={(e) => e.target.value && nav(`/live-calls/${e.target.value}`)}>
                <option value="">Switch demo scenario…</option>
                {(scenarios.data || []).map((s) => <option key={s.call_id} value={s.call_id}>{CAT_LABEL[s.category]} ({s.call_id})</option>)}
              </select>
              <Link className="btn" to={`/customers/${m.account.account_id}`}>Customer profile</Link>
              <Link className="btn" to={`/treatment?ptp=${m.ptp.ptp_id}`}>Treatment view</Link>
            </div>
          </div>

          <Card bodyClass="card-b" className="mb">
            <div className="grid g4" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))" }}>
              <div><div className="small muted">Customer / account</div><Link to={`/customers/${m.account.account_id}`} style={{ fontWeight: 600 }}>{m.account.account_id}</Link><div className="small muted">{m.account.lender?.name} · {m.account.portfolio}</div></div>
              <div><div className="small muted">DPD / bucket</div><div style={{ fontWeight: 600 }} className="num">{m.account.dpd ?? "N/A"} <span className="muted small">({m.account.bucket})</span></div></div>
              <div><div className="small muted">Outstanding</div><div style={{ fontWeight: 600 }} className="num">{fmtInr(m.account.outstanding)}</div><div className="small muted">overdue {fmtInr(m.account.overdue_at_capture)}</div></div>
              <div><div className="small muted">PTP under discussion</div><div style={{ fontWeight: 600 }} className="num">{fmtInr(m.ptp.promised_amount)} by {m.ptp.promised_date}</div><div className="small muted">agent {m.ptp.agent_id} · {m.ptp.channel}</div></div>
              <div><div className="small muted">History</div><div style={{ fontWeight: 600 }} className="num">{m.account.prior_broken ?? 0} broken / {m.account.prior_ptps ?? 0} prior PTPs</div><div className="small muted">salary day {m.account.salary_credit_day ?? "N/A"}</div></div>
              <div><div className="small muted">Call</div><div style={{ fontWeight: 600 }} className="num">{Math.round(m.call.duration_s)} s · {n} turns</div><div className="small muted">status: {status === "running" ? "in progress (replay)" : status}</div></div>
            </div>
          </Card>

          <Card className="mb" bodyClass="">
            <div className="controller">
              <strong>Call replay</strong>
              {status === "idle" && <button className="btn primary" onClick={() => setStatus("running")} disabled={busy}>Start</button>}
              {status === "running" && <button className="btn" onClick={() => setStatus("paused")}>Pause</button>}
              {status === "paused" && <button className="btn primary" onClick={() => setStatus("running")}>Resume</button>}
              {["paused", "stopped", "done"].includes(status) && <button className="btn" onClick={() => { reset(); setTimeout(() => setStatus("running"), 0); }}>Restart</button>}
              <button className="btn" onClick={() => { clearTimeout(timer.current); ctrl.current?.abort(); setStatus("stopped"); }} disabled={!["running", "paused"].includes(status)}>Stop</button>
              <div className="seg" role="group" aria-label="Replay speed">
                {[1, 2, 4].map((s) => <button key={s} className={speed === s ? "on" : ""} onClick={() => setSpeed(s)} aria-pressed={speed === s}>{s}x</button>)}
              </div>
              <div className="progress" aria-label="Transcript position"><div style={{ width: `${(pos / n) * 100}%` }} /></div>
              <span className="small muted num">Turn {pos}/{n}</span>
              <label className="row small" style={{ gap: 6 }} title={llmOk ? `Also ask ${LLM} for its view (asynchronous)` : `${LLM} is not ready`}>
                <input type="checkbox" checked={useLlm && llmOk} disabled={!llmOk || status === "running"} onChange={(e) => setUseLlm(e.target.checked)} /> {LLM}
              </label>
              <button className="btn soft" onClick={runFull} disabled={busy || status === "running"}>Score full call</button>
              <span className="small muted">{busy ? "Scoring prefix…" : status === "running" ? "Waiting for next utterance" : ""}</span>
            </div>
            {err && <div className="banner crit" style={{ margin: "0 16px 12px" }} role="alert">Prediction failed: {err.message} <button className="btn sm" onClick={() => score(Math.max(0, pos - 1))}>Retry</button></div>}
          </Card>

          <div className="live-grid">
            <Card title="Live transcript" sub="Agent/bot on the left, customer on the right. Tags show where a signal first fired." bodyClass="">
              <div className="transcript" aria-live="polite">
                {pos === 0 && <div className="state"><h4>Call not started</h4><div className="small">Press Start to replay this call through the copilot, or "Score full call".</div></div>}
                {tx.slice(0, pos).map((t, i) => (
                  <div key={i} className={`msg ${t.speaker === "customer" ? "customer" : "agent"}`}>
                    <div className="who"><span>{t.speaker === "customer" ? "Customer" : t.speaker === "bot" ? "Voice bot" : "Agent"}</span><span className="num">{(t.start_ms / 1000).toFixed(1)}s{t.speaker === "customer" ? ` · pause ${t.pause_before_ms} ms · ASR ${t.asr_confidence?.toFixed(2)}` : ""}</span></div>
                    <div>{t.text}</div>
                    {tagsFor(i).length > 0 && (
                      <div className="tags">{tagsFor(i).map((e) => <span key={e.signal} className={`tag ${SIG_TAG[e.signal][1]}`}>{SIG_TAG[e.signal][0]}</span>)}</div>
                    )}
                  </div>
                ))}
              </div>
            </Card>

            {!a && <Card className="intel" title="Copilot now"><div className="state small">The credibility score, promise type and suggested next line appear here after the first utterance is scored.</div></Card>}
            {a && (
              <Card className="intel now" title="Copilot now" sub={`After turn ${a.turn + 1} of ${a.n_turns}`}
                right={<div className="row" style={{ gap: 6 }}>
                  {heardTurn !== null && <span className="badge measured" title="Score stable and a commitment or third-party event has occurred">Heard enough · turn {heardTurn + 1}</span>}
                  <StatusBadge s="MODEL OUTPUT" />
                </div>}>
                {a.hardship_safeguard && <div className="banner warn compact" role="status"><strong>HARDSHIP SAFEGUARD ACTIVE</strong><span>No aggressive treatment: restructuring, smaller amount or post-salary date.</span></div>}
                {a.privacy_safeguard && <div className="banner info compact" role="status"><strong>PRIVACY SAFEGUARD ACTIVE</strong><span>Third party on the line: do not disclose the debt.</span></div>}

                <div className="ask-box">
                  <div className="between">
                    <span className="ask-title">Say next <span className="muted small">· {a.treatment.utterance_kind}</span></span>
                    <span className="small">Policy check: {a.treatment.policy_check === "pass" ? <b style={{ color: "#15803d" }}>pass</b> : <b style={{ color: "#dc2626" }}>blocked: {a.treatment.compliance_issues.join(", ")}</b>}</span>
                  </div>
                  {a.treatment.ask?.length > 0 && (
                    <div className="ask-chips">
                      {a.treatment.ask.map((k, i) => (
                        <div key={i} className={`ask-chip ${k.tone || ""}`}><span>{k.tag}</span><b>{k.text}</b></div>
                      ))}
                    </div>
                  )}
                  <div className="small muted" style={{ marginTop: 10 }}>Say it like this</div>
                  <div className="script big-script">{a.treatment.next_utterance}</div>
                </div>

                <div className="now-top" style={{ marginTop: 14 }}>
                  <div style={{ textAlign: "center" }}>
                    <Radial value={a.credibility.p_keep} label="credibility" size={96} />
                    <div className={`small risk-${a.credibility.risk}`} style={{ fontWeight: 600, textTransform: "capitalize" }}>{a.credibility.risk} risk</div>
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="small muted">Promise type (decision, after guardrails)</div>
                    <div className="now-cat"><CategoryPill c={a.final_category} /></div>
                    <div className="small" style={{ display: "grid", gap: 4 }}>
                      <div className="row" style={{ gap: 6 }}><span className="muted">LightGBM</span><CategoryPill c={a.tabular_category.category} conf={a.tabular_category.confidence} /></div>
                      <div className="row" style={{ gap: 6 }}><span className="muted">{LLM}</span>
                        {L
                          ? <><CategoryPill c={L.category} conf={L.confidence} /><span className="muted">{L.category === a.tabular_category.category ? "agrees" : "disagrees"}{L.turn < a.turn ? ` · as of turn ${L.turn + 1}` : ""}</span></>
                          : <span className="muted">{!llmOk ? "unavailable" : !useLlm ? "off" : "waiting…"}</span>}
                        {llmWorking && <span className="muted small">analysing…</span>}
                      </div>
                      {L?.reasons?.length > 0 && <div className="muted" style={{ fontStyle: "italic" }}>{LLM}: {L.reasons[0]}</div>}
                      <div className="muted num">P(kept) from LightGBM in-call model · tabular {a.latency_ms.tabular} ms{L?.latency_ms ? ` · ${LLM} ${L.latency_ms} ms` : ""}</div>
                    </div>
                  </div>
                </div>


                <div className="now-grid">
                  <div><span>Action</span><b>{QUEUE_LABEL[a.treatment.queue] || a.treatment.queue}</b></div>
                  <div><span>Parking</span><b>{a.treatment.parking_days} day(s)</b></div>
                  <div><span>Reminders</span><b>{a.treatment.reminders.join(", ") || "None"}</b></div>
                  <div><span>Follow-up</span><b>{a.treatment.follow_up}</b></div>
                </div>

                <div className="now-commit">
                  <span className={`pill own-${OWN[a.commitment.ownership][1]}`}>{OWN[a.commitment.ownership][0]}</span>
                  <span><span className="muted">Date</span> {a.commitment.latest.date ? `${a.commitment.latest.date.value} (by ${a.commitment.first.date.by})` : "not stated"}</span>
                  <span className="num"><span className="muted">Amount</span> {a.commitment.latest.amount ? `${fmtInr(a.commitment.latest.amount.value)} (by ${a.commitment.first.amount.by})` : "not stated"}</span>
                  <span><span className="muted">Confirmation</span> {a.commitment.borrower_explicit_confirmation ? "explicit" : "none"}</span>
                </div>

                {a.evidence.length > 0 && (
                  <div className="tags" style={{ marginTop: 10 }}>
                    {a.evidence.map((e) => (
                      <span key={e.signal} className={`tag ${{ "+": "pos", "-": "neg", "!": "alert", "~": "" }[e.polarity]}`} title={e.quote ? `turn ${e.first_turn + 1}: "${e.quote}"` : undefined}>
                        {e.text}{e.count > 1 ? ` ×${e.count}` : ""}
                      </span>
                    ))}
                  </div>
                )}
                {a.guardrails.length > 0 && <div className="small muted" style={{ marginTop: 8 }}>Guardrails applied: {a.guardrails.join("; ")}</div>}
              </Card>
            )}
          </div>

          {a && (
            <div className="grid g2 mt">
              <Card title="Score timeline" sub={`Credibility after every utterance; ${LLM} confidence where it has answered`} right={<StatusBadge s="MODEL OUTPUT" />}>
                {timeline.length > 1 ? (
                  <ResponsiveContainer width="100%" height={210}>
                    <LineChart data={timeline} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="turn" />
                      <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} />
                      <Tooltip formatter={(v, nme) => [v === null ? "N/A" : fmtPct(v, 0), nme]} labelFormatter={(t) => `Turn ${t}`} />
                      <Legend />
                      <Line name="Credibility" dataKey="credibility" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                      <Line name={`${LLM} category conf.`} dataKey="llm_conf" stroke="#93c5fd" strokeWidth={2} dot={{ r: 2 }} connectNulls isAnimationActive={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : <div className="small muted">The timeline appears after two scored utterances.</div>}
              </Card>

              <Card title="Evidence detail" sub="Each signal with the first utterance that triggered it">
                {a.evidence.length === 0 ? <div className="small muted">No signals yet.</div> : a.evidence.map((e) => (
                  <div key={e.signal} className="ev">
                    <span className={`ic ${{ "+": "pos", "-": "neg", "!": "alert", "~": "neu" }[e.polarity]}`}>{e.polarity === "~" ? "•" : e.polarity}</span>
                    <div>
                      <div>{e.text}{e.count > 1 ? <span className="muted"> ×{e.count}</span> : ""}</div>
                      {e.quote && <div className="q">turn {e.first_turn + 1}: "{e.quote}"</div>}
                    </div>
                  </div>
                ))}
                <div className="small muted mt">Top score drivers (SHAP, log-odds):</div>
                {a.drivers.map((d) => (
                  <div key={d.feature} className="between small" style={{ padding: "3px 0" }}>
                    <span>{d.label}</span><span className="num" style={{ color: d.contribution > 0 ? "#15803d" : "#b45309" }}>{d.contribution > 0 ? "+" : ""}{d.contribution.toFixed(2)}</span>
                  </div>
                ))}
              </Card>

              <Card title={`${LLM} raw draft`} sub={`Free-generated text from ${LLM}, for transparency only; the agent always sees the approved script above`}>
                <div className="row">
                  <button className="btn sm" onClick={runGen} disabled={!llmOk || pos === 0 || gen?.loading}>Generate raw {LLM} draft</button>
                  {!llmOk && <span className="small muted">{LLM} unavailable</span>}
                </div>
                {gen?.loading && <div className="small muted mt">Generating with {LLM}…</div>}
                {gen?.error && <div className="banner crit mt">{gen.error.message}</div>}
                {gen && gen.available && (
                  <div className="banner mt" style={{ display: "block" }}>
                    <div className="between"><strong className="small">Raw {LLM} output (not shown to agent)</strong><span className="small num">{gen.tokens ? `${gen.tokens} tokens · ` : ""}{gen.latency_ms} ms <StatusBadge s="MEASURED" /></span></div>
                    <div className="mono" style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{gen.raw}</div>
                    <div className="small muted" style={{ marginTop: 6 }}>{gen.note} Raw-text compliance: {gen.compliance_issues_raw?.length ? gen.compliance_issues_raw.join(", ") : "no issues"}.</div>
                  </div>
                )}
                {gen && gen.available === false && <div className="mt"><Unavailable title={`${LLM} unavailable`}>{gen.reason}</Unavailable></div>}
              </Card>

              {a.is_final ? (
                <Card title="After the call: ground truth" sub={a.ground_truth.note} right={<StatusBadge s="ACTUAL" />}>
                  <dl className="kv">
                    <dt>Ground-truth category</dt><dd><CategoryPill c={a.ground_truth.category} /></dd>
                    <dt>Payment outcome</dt><dd>{STATUS_LABEL[a.ground_truth.outcome] || a.ground_truth.outcome}</dd>
                    {a.audit_id && <><dt>Audit record</dt><dd><Link to={`/audit/${a.audit_id}`}>{a.audit_id}</Link></dd></>}
                  </dl>
                </Card>
              ) : (
                <Card title="After the call: ground truth" right={<StatusBadge s="ACTUAL" />}>
                  <div className="small muted">Revealed only after the last utterance, so it cannot influence the live prediction.</div>
                </Card>
              )}
            </div>
          )}
        </>
      )}
    </Async>
  );
}
