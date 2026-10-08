import { Link, useParams } from "react-router-dom";
import { auditApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, Card, CategoryPill, PageHead, QUEUE_LABEL, StatusBadge, fmtDateTime, fmtPct } from "../components/ui";

export default function AuditDetail() {
  const { auditId } = useParams();
  const q = useApi((o) => auditApi.get(auditId, o), [auditId]);
  return (
    <Async q={q} rows={8}>
      {(r) => (
        <>
          <PageHead title={`Audit record ${r.audit_id}`} desc={`${fmtDateTime(r.timestamp)} · ${r.kind}`} right={<Link className="btn" to="/audit">Back to audit trail</Link>} />
          <div className="grid g2">
            <Card title="Input / context" right={<StatusBadge s="ACTUAL" />}>
              <dl className="kv">
                <dt>Customer</dt><dd><Link to={`/customers/${r.account_id}`}>{r.account_id}</Link></dd>
                <dt>PTP</dt><dd><Link to={`/treatment?ptp=${r.ptp_id}`}>{r.ptp_id}</Link></dd>
                <dt>Conversation</dt><dd><Link to={`/live-calls/${r.call_id}`}>{r.call_id}</Link></dd>
                <dt>Transcript prefix</dt><dd>turns 1–{r.turn + 1} of {r.n_turns}</dd>
                <dt>Model versions</dt><dd className="mono">{r.model_version.tabular}<br />{r.model_version.qwen}</dd>
                <dt>Latency</dt><dd className="num">tabular {r.latency_ms.tabular} ms{r.latency_ms.qwen ? ` · Qwen ${r.latency_ms.qwen} ms` : ""}</dd>
              </dl>
            </Card>
            <Card title="Model output" right={<StatusBadge s="MODEL OUTPUT" />}>
              <dl className="kv">
                <dt>Credibility P(kept)</dt><dd>{fmtPct(r.p_keep)}</dd>
                <dt>Tabular category</dt><dd><CategoryPill c={r.tabular_category} conf={r.tabular_confidence} /></dd>
                <dt>Qwen category</dt><dd>{r.qwen_available ? <CategoryPill c={r.qwen_category} conf={r.qwen_confidence} /> : "Unavailable"}</dd>
                <dt>Final category</dt><dd><CategoryPill c={r.final_category} /></dd>
              </dl>
              {r.extra?.raw && <div className="banner mt" style={{ display: "block" }}><b className="small">Raw Qwen draft (not shown to agent)</b><div className="mono" style={{ marginTop: 4 }}>{r.extra.raw}</div></div>}
            </Card>
            <Card title="Evidence">
              {r.evidence.length ? <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>{r.evidence.map((e) => <li key={e}>{e}</li>)}</ul> : <span className="muted">None</span>}
            </Card>
            <Card title="Treatment, safeguards and policy">
              <dl className="kv">
                <dt>Treatment</dt><dd>{QUEUE_LABEL[r.treatment.queue] || r.treatment.queue} · park {r.treatment.parking_days} day(s)</dd>
                <dt>Follow-up</dt><dd>{r.treatment.follow_up}</dd>
                <dt>Script shown</dt><dd className="small">{r.treatment.next_utterance}</dd>
                <dt>Guardrails</dt><dd>{r.guardrails.length ? r.guardrails.join("; ") : "None"}</dd>
                <dt>Hardship safeguard</dt><dd>{r.hardship_safeguard ? "Active" : "Not active"}</dd>
                <dt>Privacy safeguard</dt><dd>{r.privacy_safeguard ? "Active" : "Not active"}</dd>
                <dt>Policy check</dt><dd>{r.policy_check}{r.treatment.compliance_issues.length ? `: ${r.treatment.compliance_issues.join(", ")}` : ""}</dd>
                <dt>Audit status</dt><dd>{r.audit_status}</dd>
              </dl>
            </Card>
          </div>
        </>
      )}
    </Async>
  );
}
