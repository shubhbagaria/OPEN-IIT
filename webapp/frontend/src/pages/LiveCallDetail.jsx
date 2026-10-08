import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { liveCallsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { qwenReady, useHealth } from "../components/Health";
import { Async, CAT_LABEL, Card, CategoryPill, Meter, QUEUE_LABEL, Radial, StatusBadge, STATUS_LABEL, Unavailable, fmtInr, fmtPct } from "../components/ui";

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
  const [useQwen, setUseQwen] = useState(true);
  const [gen, setGen] = useState(null);
  const timer = useRef(null);
  const ctrl = useRef(null);
  const statusRef = useRef(status);
  statusRef.current = status;
  const tx = meta.data?.transcript || [];
  const n = tx.length;
  const qwenOk = qwenReady(health);

  const reset = useCallback(() => {
    clearTimeout(timer.current);
    ctrl.current?.abort();
    setStatus("idle"); setPos(0); setAssess({}); setErr(null); setGen(null); setBusy(false);
  }, []);
  useEffect(() => reset, [callId, reset]);

  const score = useCallback(async (turn) => {
    ctrl.current = new AbortController();
    setBusy(true);
    try {
      const a = await liveCallsApi.step(callId, { turn, qwen: useQwen && qwenOk }, { signal: ctrl.current.signal });
      setAssess((m) => ({ ...m, [turn]: a }));
      setErr(null);
      return a;
    } catch (e) {
      if (e.name !== "AbortError") { setErr(e); setStatus("paused"); }
      return null;
    } finally {
      setBusy(false);
    }
  }, [callId, useQwen, qwenOk]);

  // simulation loop: reveal next turn after its real recorded gap / speed, then score the prefix
  useEffect(() => {
    if (status !== "running") return;
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
  }, [status, pos, n, speed]);

  const lastTurn = Math.max(-1, ...Object.keys(assess).map(Number));
  const a = lastTurn >= 0 ? assess[lastTurn] : null;
  const timeline = Object.values(assess).sort((x, y) => x.turn - y.turn).map((x) => ({
    turn: x.turn + 1, credibility: x.credibility.p_keep, qwen_conf: x.qwen?.available ? x.qwen.confidence : null,
    tab_conf: x.tabular_category.confidence,
  }));

  const runFull = async () => {
    reset();
    setBusy(true);
    try {
      const r = await liveCallsApi.predict(callId);
      setPos(n); setAssess({ [r.turn]: r }); setStatus("done");
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
              <label className="row small" style={{ gap: 6 }} title={qwenOk ? "Also score each prefix with the fine-tuned Qwen SLM" : "Qwen SLM is not ready"}>
                <input type="checkbox" checked={useQwen && qwenOk} disabled={!qwenOk || status === "running"} onChange={(e) => setUseQwen(e.target.checked)} /> Qwen SLM
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

            <div className="grid">
              {!a && <Card className="intel" title="Copilot intelligence"><div className="state small">Model output appears after the first utterance is scored.</div></Card>}
              {a && (
                <>
                  {a.hardship_safeguard && <div className="banner warn" role="status"><strong>HARDSHIP SAFEGUARD ACTIVE</strong><span>Hardship signals present: no aggressive treatment; restructuring or smaller amount / post-salary date. {a.guardrails.filter((g) => g.includes("hardship") || g.includes("vulnerable")).join("; ")}</span></div>}
                  {a.privacy_safeguard && <div className="banner info" role="status"><strong>PRIVACY SAFEGUARD ACTIVE</strong><span>A third party is on the line: do not disclose the debt; confirm with the borrower before parking.</span></div>}
                  <Card className="intel" title="Copilot intelligence" sub={`After turn ${a.turn + 1} of ${a.n_turns}`}
                    right={<StatusBadge s="MODEL OUTPUT" />}>
                    <div className="row" style={{ alignItems: "flex-start", gap: 20 }}>
                      <div style={{ textAlign: "center" }}>
                        <Radial value={a.credibility.p_keep} label="credibility" />
                        <div className="small muted" style={{ maxWidth: 130 }}>P(kept), LightGBM in-call, calibrated</div>
                      </div>
                      <div style={{ flex: 1, minWidth: 220 }}>
                        <div className="small muted">Decision category (after guardrails)</div>
                        <div style={{ margin: "4px 0 10px" }}><CategoryPill c={a.final_category} /></div>
                        <dl className="kv">
                          <dt>Tabular model</dt><dd><CategoryPill c={a.tabular_category.category} conf={a.tabular_category.confidence} /></dd>
                          <dt>Qwen SLM</dt>
                          <dd>{a.qwen.available ? (
                            <span className="row" style={{ gap: 6 }}><CategoryPill c={a.qwen.category} conf={a.qwen.confidence} />
                              <span className="small muted">{a.qwen.category === a.tabular_category.category ? "agrees" : "disagrees"} · {a.qwen.latency_ms} ms</span></span>
                          ) : <span className="muted small">{a.qwen.status === "skipped" ? "Not run for this step" : `Unavailable: ${a.qwen.reason}`}</span>}</dd>
                          <dt>Risk</dt><dd className={`risk-${a.credibility.risk}`} style={{ textTransform: "capitalize" }}>{a.credibility.risk}</dd>
                          <dt>Latency</dt><dd className="num">tabular {a.latency_ms.tabular} ms{a.latency_ms.qwen ? ` · Qwen ${a.latency_ms.qwen} ms` : ""} <StatusBadge s="MEASURED" /></dd>
                        </dl>
                      </div>
                    </div>
                    {timeline.length > 1 && (
                      <div className="mt">
                        <div className="small muted">Score timeline</div>
                        <ResponsiveContainer width="100%" height={150}>
                          <LineChart data={timeline} margin={{ top: 6, right: 8, left: -18, bottom: 0 }}>
                            <CartesianGrid vertical={false} />
                            <XAxis dataKey="turn" />
                            <YAxis domain={[0, 1]} tickFormatter={(v) => `${Math.round(v * 100)}`} />
                            <Tooltip formatter={(v, nme) => [v === null ? "N/A" : fmtPct(v, 0), nme]} labelFormatter={(t) => `Turn ${t}`} />
                            <Legend />
                            <Line name="Credibility" dataKey="credibility" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
                            <Line name="Qwen category conf." dataKey="qwen_conf" stroke="#93c5fd" strokeWidth={2} dot={{ r: 2 }} connectNulls isAnimationActive={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </Card>

                  <Card title="Commitment" sub="Who said the date and amount first (read from the transcript)">
                    <dl className="kv">
                      <dt>Ownership</dt><dd><span className={`pill`}>{OWN[a.commitment.ownership][0]}</span></dd>
                      <dt>Date</dt><dd>{a.commitment.latest.date ? `${a.commitment.latest.date.value} (first by ${a.commitment.first.date.by}, turn ${a.commitment.first.date.turn + 1})` : "Not stated yet"}</dd>
                      <dt>Amount</dt><dd className="num">{a.commitment.latest.amount ? `${fmtInr(a.commitment.latest.amount.value)} (first by ${a.commitment.first.amount.by}, turn ${a.commitment.first.amount.turn + 1})` : "Not stated yet"}</dd>
                      <dt>Borrower confirmation</dt><dd>{a.commitment.borrower_explicit_confirmation ? "Explicit (firm language or part-payment offer)" : "None explicit"}</dd>
                    </dl>
                  </Card>

                  <Card title="Evidence" sub="Signals detected so far; quotes are the first utterance that triggered each one">
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

                  <Card title="Treatment recommendation" sub="From ptp_copilot.policy: guardrails, approved script, compliance filter, parking engine" right={<StatusBadge s="MODEL OUTPUT" />}>
                    <div className="small muted">Suggested next line ({a.treatment.utterance_kind})</div>
                    <div className="script mt" style={{ marginTop: 6 }}>{a.treatment.next_utterance}</div>
                    <div className="row small" style={{ marginTop: 6 }}>
                      Policy check: {a.treatment.policy_check === "pass" ? <span style={{ color: "#15803d", fontWeight: 600 }}>pass</span> : <span style={{ color: "#dc2626", fontWeight: 600 }}>blocked: {a.treatment.compliance_issues.join(", ")}</span>}
                    </div>
                    <dl className="kv mt">
                      <dt>Action</dt><dd>{QUEUE_LABEL[a.treatment.queue] || a.treatment.queue}</dd>
                      <dt>Parking</dt><dd>{a.treatment.parking_days} day(s)</dd>
                      <dt>Reminders</dt><dd>{a.treatment.reminders.join(", ") || "None"}</dd>
                      <dt>Follow-up</dt><dd>{a.treatment.follow_up}</dd>
                      {a.guardrails.length > 0 && <><dt>Guardrails</dt><dd>{a.guardrails.join("; ")}</dd></>}
                    </dl>
                    <div className="row mt">
                      <button className="btn sm" onClick={runGen} disabled={!qwenOk || pos === 0 || gen?.loading}>Generate raw Qwen draft</button>
                      {!qwenOk && <span className="small muted">Qwen SLM unavailable</span>}
                    </div>
                    {gen?.loading && <div className="small muted mt">Generating with Qwen…</div>}
                    {gen?.error && <div className="banner crit mt">{gen.error.message}</div>}
                    {gen && gen.available && (
                      <div className="banner mt" style={{ display: "block" }}>
                        <div className="between"><strong className="small">Raw Qwen output (not shown to agent)</strong><span className="small num">{gen.tokens} tokens · {gen.latency_ms} ms <StatusBadge s="MEASURED" /></span></div>
                        <div className="mono" style={{ marginTop: 6, whiteSpace: "pre-wrap" }}>{gen.raw}</div>
                        <div className="small muted" style={{ marginTop: 6 }}>{gen.note} Raw-text compliance: {gen.compliance_issues_raw?.length ? gen.compliance_issues_raw.join(", ") : "no issues"}.</div>
                      </div>
                    )}
                    {gen && gen.available === false && <div className="mt"><Unavailable title="Qwen SLM unavailable">{gen.reason}</Unavailable></div>}
                  </Card>

                  {a.is_final && (
                    <Card title="After the call: ground truth" sub={a.ground_truth.note} right={<StatusBadge s="ACTUAL" />}>
                      <dl className="kv">
                        <dt>Ground-truth category</dt><dd><CategoryPill c={a.ground_truth.category} /></dd>
                        <dt>Payment outcome</dt><dd>{STATUS_LABEL[a.ground_truth.outcome] || a.ground_truth.outcome}</dd>
                        {a.audit_id && <><dt>Audit record</dt><dd><Link to={`/audit/${a.audit_id}`}>{a.audit_id}</Link></dd></>}
                      </dl>
                    </Card>
                  )}
                </>
              )}
            </div>
          </div>
        </>
      )}
    </Async>
  );
}
