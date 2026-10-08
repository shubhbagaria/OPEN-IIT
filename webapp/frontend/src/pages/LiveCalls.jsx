import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { liveCallsApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, CATS, CAT_LABEL, Card, CategoryPill, DataTable, PageHead, Pager, QUEUE_LABEL, StatusBadge, STATUS_LABEL, fmtInr, fmtPct } from "../components/ui";

export default function LiveCalls() {
  const nav = useNavigate();
  const [sp, setSp] = useSearchParams();
  const get = (k, d = "") => sp.get(k) ?? d;
  const [qText, setQText] = useState(get("q"));
  const params = {
    q: get("q"), category: get("category"), risk: get("risk"), status: get("status"), split: get("split"), lang: get("lang"),
    sort: get("sort", "captured_ts"), order: get("order", "desc"), page: Number(get("page", "1")), size: 25,
  };
  const set = (patch) => {
    const n = new URLSearchParams(sp);
    Object.entries(patch).forEach(([k, v]) => (v === "" || v === undefined ? n.delete(k) : n.set(k, v)));
    if (!("page" in patch)) n.delete("page");
    setSp(n);
  };
  useEffect(() => {
    const t = setTimeout(() => qText !== params.q && set({ q: qText }), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qText]);

  const list = useApi((o) => liveCallsApi.list(params, o), [sp.toString()]);
  const sc = useApi((o) => liveCallsApi.scenarios(o));
  const onSort = (key) => set({ sort: key, order: params.sort === key && params.order === "desc" ? "asc" : "desc" });

  const cols = [
    { key: "account_id", label: "Customer / account", sortable: true, render: (r) => <div><div style={{ fontWeight: 600 }}>{r.account_id}</div><div className="small muted">{r.call_id} · {r.preferred_language}</div></div> },
    { key: "dpd_start", label: "DPD", sortable: true, num: true },
    { key: "outstanding", label: "Outstanding", sortable: true, num: true, render: (r) => fmtInr(r.outstanding) },
    { key: "duration_s", label: "Duration", sortable: true, num: true, render: (r) => `${Math.floor(r.duration_s / 60)}:${String(Math.round(r.duration_s % 60)).padStart(2, "0")}` },
    { key: "promised_amount", label: "PTP", sortable: true, num: true, render: (r) => <div>{fmtInr(r.promised_amount)}<div className="small muted">due {r.promised_date?.slice(0, 10)}</div></div> },
    { key: "confidence_score", label: "Credibility", sortable: true, num: true, render: (r) => fmtPct(r.confidence_score, 0) },
    { key: "predicted_category", label: "Category", render: (r) => <CategoryPill c={r.predicted_category} /> },
    { key: "risk", label: "Risk", render: (r) => <span className={`risk-${r.risk}`} style={{ fontWeight: 600, textTransform: "capitalize" }}>{r.risk}</span> },
    { key: "queue", label: "Recommended action", render: (r) => QUEUE_LABEL[r.queue] || r.queue },
    { key: "status", label: "Actual outcome", render: (r) => <span className="muted">{STATUS_LABEL[r.status]}</span> },
  ];

  return (
    <>
      <PageHead title="Live Calls"
        desc="Recorded PTP calls that can be replayed through the copilot. Open a call to stream it utterance by utterance through the real models." />
      <Card title="Demo scenarios" sub="Held-out test-split calls, one per ground-truth category (human-annotated where available). Opening one starts in replay mode."
        right={<StatusBadge s="LIVE DEMO SIMULATION" />} className="mb">
        <Async q={sc} rows={2}>
          {(rows) => (
            <div className="row">
              {rows.map((s) => (
                <button key={s.call_id} className="btn soft" onClick={() => nav(`/live-calls/${s.call_id}`)}>
                  {CAT_LABEL[s.category]} <span className="small muted">· {s.n_turns} turns · {s.language}{s.label_source === "human_annotation" ? " · human label" : ""}</span>
                </button>
              ))}
            </div>
          )}
        </Async>
      </Card>
      <Card bodyClass="" title="All replayable calls"
        sub={list.data?.provenance}
        right={<StatusBadge s="STORED MODEL OUTPUT" />}>
        <div className="toolbar" style={{ padding: "12px 16px 0" }}>
          <input className="input" placeholder="Search account, call, PTP or agent ID" value={qText} onChange={(e) => setQText(e.target.value)} aria-label="Search calls" />
          <select className="select" value={params.category} onChange={(e) => set({ category: e.target.value })} aria-label="Category">
            <option value="">All categories</option>
            {CATS.map((c) => <option key={c} value={c}>{CAT_LABEL[c]}</option>)}
          </select>
          <select className="select" value={params.risk} onChange={(e) => set({ risk: e.target.value })} aria-label="Risk">
            <option value="">All risk</option><option value="high">High risk (&lt;25%)</option><option value="medium">Medium (25–50%)</option><option value="low">Low (≥50%)</option>
          </select>
          <select className="select" value={params.status} onChange={(e) => set({ status: e.target.value })} aria-label="Outcome">
            <option value="">All outcomes</option>
            {Object.entries(STATUS_LABEL).filter(([k]) => k !== "pending").map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <select className="select" value={params.lang} onChange={(e) => set({ lang: e.target.value })} aria-label="Language">
            <option value="">All languages</option><option value="hinglish">Hinglish</option><option value="kanglish">Kanglish</option><option value="english">English</option>
          </select>
          <select className="select" value={params.split} onChange={(e) => set({ split: e.target.value })} aria-label="Data split">
            <option value="">All splits</option><option value="test">Held-out test</option><option value="validation">Validation</option><option value="train">Train</option>
          </select>
          <button className="btn" onClick={() => { setQText(""); setSp(new URLSearchParams()); }}>Clear filters</button>
        </div>
        <div style={{ marginTop: 12 }}>
          <Async q={list} rows={8}>
            {(d) => (
              <>
                <DataTable columns={cols} rows={d.rows} rowKey="call_id" sort={{ key: params.sort, dir: params.order }} onSort={onSort}
                  onRowClick={(r) => nav(`/live-calls/${r.call_id}`)} empty="Try clearing filters." />
                <Pager page={d.page} size={d.size} total={d.total} onPage={(p) => set({ page: p })} />
              </>
            )}
          </Async>
        </div>
      </Card>
    </>
  );
}
