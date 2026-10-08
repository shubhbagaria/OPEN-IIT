import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { auditApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CATS, CAT_LABEL, Card, CategoryPill, DataTable, Kpi, PageHead, StatusBadge, fmtDateTime, fmtPct, useClientSort } from "../components/ui";

const KIND = { live_simulation_final: "Live simulation (final turn)", full_call_prediction: "Full-call prediction", treatment_decision: "Treatment decision", slm_generation: "Qwen draft generation" };

export default function Audit() {
  const nav = useNavigate();
  const [f, setF] = useState({ q: "", kind: "", status: "", category: "" });
  const q = useApi((o) => auditApi.list(f, o), [JSON.stringify(f)]);
  const s = useClientSort(q.data?.rows || [], { key: "timestamp", dir: "desc" });
  const b = q.data?.batch_audit;
  return (
    <>
      <PageHead title="Audit & Compliance" desc="Every decision the application produces is appended to outputs/app_audit.jsonl with inputs, model versions, outputs, safeguards and policy status." />
      {b && (
        <Card className="mb" title="Batch compliance audit over all 3,696 PTPs" sub="From the offline pipeline (results.json)" right={<StatusBadge s="MEASURED" />}>
          <div className="grid g4">
            <Kpi label="Hardship PTPs routed to aggressive treatment" value={`${b.hardship_routed_to_aggressive_after_guardrails} / ${b.hardship_signal_ptps}`} note={`${b.hardship_softened_by_guardrail} softened by a guardrail`} />
            <Kpi label="Third-party scripts with debt disclosure" value={`${b.third_party_utterances_with_debt_disclosure} / ${b.third_party_ptps}`} note="DPDP / RBI FPC" />
            <Kpi label="Scripts with threatening language" value={b.any_utterance_with_banned_language} note="Deterministic compliance filter" />
            <Kpi label="Historical agent disclosure to third parties" value={fmtPct(b.historical_agent_disclosure_after_3p_identified.share, 0)}
              note={`${b.historical_agent_disclosure_after_3p_identified.calls_with_post_identification_disclosure} of ${b.historical_agent_disclosure_after_3p_identified.calls_with_third_party} calls (baseline, ACTUAL)`} />
          </div>
        </Card>
      )}
      <Card bodyClass="" title="Decision audit trail" sub="Records created by this application" right={<StatusBadge s="MODEL OUTPUT" />}>
        <div className="toolbar" style={{ padding: "12px 16px 0" }}>
          <input className="input" placeholder="Search audit, PTP, call or account ID" value={f.q} onChange={(e) => setF({ ...f, q: e.target.value })} aria-label="Search audit" />
          <select className="select" value={f.kind} onChange={(e) => setF({ ...f, kind: e.target.value })} aria-label="Record type">
            <option value="">All record types</option>{Object.entries(KIND).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select className="select" value={f.status} onChange={(e) => setF({ ...f, status: e.target.value })} aria-label="Audit status">
            <option value="">All statuses</option><option value="clear">Clear</option><option value="flagged">Flagged (guardrail or policy)</option>
          </select>
          <select className="select" value={f.category} onChange={(e) => setF({ ...f, category: e.target.value })} aria-label="Category">
            <option value="">All categories</option>{CATS.map((c) => <option key={c} value={c}>{CAT_LABEL[c]}</option>)}
          </select>
          <button className="btn" onClick={() => setF({ q: "", kind: "", status: "", category: "" })}>Clear filters</button>
          <button className="btn" onClick={q.retry}>Refresh</button>
        </div>
        <div style={{ marginTop: 12 }}>
          <Async q={q} rows={6}>
            {() => (
              <DataTable rowKey="audit_id" rows={s.rows} sort={s.sort} onSort={s.onSort} onRowClick={(r) => nav(`/audit/${r.audit_id}`)}
                empty="No decisions recorded yet. Run a live call, a full-call prediction or a treatment decision."
                columns={[
                  { key: "timestamp", label: "Timestamp", sortable: true, render: (r) => fmtDateTime(r.timestamp) },
                  { key: "account_id", label: "Customer", sortable: true },
                  { key: "call_id", label: "Conversation", sortable: true, render: (r) => <div>{r.call_id}<div className="small muted">{r.ptp_id}</div></div> },
                  { key: "kind", label: "Type", render: (r) => <span className="small">{KIND[r.kind] || r.kind}</span> },
                  { key: "model", label: "Model version", render: (r) => <span className="mono">{r.model_version?.qwen}</span> },
                  { key: "qwen_confidence", label: "Qwen", sortable: true, render: (r) => r.qwen_available ? <CategoryPill c={r.qwen_category} conf={r.qwen_confidence} /> : <span className="muted small">N/A</span> },
                  { key: "p_keep", label: "Credibility", sortable: true, num: true, render: (r) => fmtPct(r.p_keep, 0) },
                  { key: "final_category", label: "Final category", sortable: true, render: (r) => <CategoryPill c={r.final_category} conf={r.tabular_confidence} /> },
                  { key: "treatment", label: "Treatment", render: (r) => <span className="small">{r.treatment.queue}</span> },
                  { key: "hardship_safeguard", label: "Hardship", render: (r) => r.hardship_safeguard ? <span className="badge projected">Active</span> : <span className="muted small">No</span> },
                  { key: "policy_check", label: "Policy", render: (r) => r.policy_check },
                  { key: "audit_status", label: "Status", sortable: true, render: (r) => <span className={`badge ${r.audit_status === "clear" ? "measured" : "projected"}`}>{r.audit_status}</span> },
                ]} />
            )}
          </Async>
        </div>
      </Card>
    </>
  );
}
