import { useMemo, useState } from "react";

export const CAT_LABEL = {
  genuine_feasible: "Genuine & Feasible",
  genuine_infeasible: "Genuine but Infeasible",
  escape: "Escape Promise",
  third_party: "Third-Party Promise",
  repeat_promiser: "Repeat Promiser",
  agent_recorded_or_pushed: "Agent-Recorded / Pushed",
};
export const CAT_COLOR = {
  genuine_feasible: "#16a34a",
  genuine_infeasible: "#d97706",
  escape: "#d97706",
  third_party: "#3b82f6",
  repeat_promiser: "#c2410c",
  agent_recorded_or_pushed: "#dc2626",
};
export const CATS = Object.keys(CAT_LABEL);
export const CAT_SHORT = { genuine_feasible: "Feasible", genuine_infeasible: "Infeasible", escape: "Escape", third_party: "Third party", repeat_promiser: "Repeat", agent_recorded_or_pushed: "Agent-pushed" };
export const STATUS_LABEL = { kept_on_time: "Kept on time", kept_late: "Kept late (cure)", partial: "Partial", broken: "Broken", pending: "Pending" };
export const QUEUE_LABEL = {
  park: "Park until due date", park_with_check: "Park with pre-due check", early_followup: "Early follow-up (24–48h)",
  restructuring_review: "Restructuring review", hardship_desk_human: "Hardship desk (human)", senior_caller: "Senior caller",
  rpc_confirmation: "Confirm with borrower", integrity_review: "Integrity review",
};

export const fmtPct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "N/A" : `${(v * 100).toFixed(d)}%`);
export const fmtInt = (v) => (v === null || v === undefined ? "N/A" : Number(v).toLocaleString("en-IN"));
export function fmtInr(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "N/A";
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return `₹${Math.round(v).toLocaleString("en-IN")}`;
}
export const fmtDate = (s) => (s ? new Date(s).toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" }) : "N/A");
export const fmtDateTime = (s) => (s ? new Date(s).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" }) : "N/A");

export function StatusBadge({ s }) {
  const cls = {
    ACTUAL: "actual", "MODEL OUTPUT": "model", MEASURED: "measured", SIMULATED: "simulated", PROTOTYPE: "prototype",
    TARGET: "target", PROJECTED: "projected", ILLUSTRATIVE: "illustrative", UNAVAILABLE: "unavailable",
    "STORED MODEL OUTPUT": "model", "LIVE DEMO SIMULATION": "simulated",
  }[s] || "";
  return <span className={`badge ${cls}`} title={BADGE_HELP[s]}>{s}</span>;
}
const BADGE_HELP = {
  ACTUAL: "Computed directly from the supplied data",
  "MODEL OUTPUT": "Produced by the trained models",
  "STORED MODEL OUTPUT": "Out-of-fold model output stored by the offline pipeline",
  MEASURED: "Measured in offline evaluation on held-out data",
  PROJECTED: "Projection that depends on stated assumptions; requires a randomised trial",
  "LIVE DEMO SIMULATION": "Recorded call replayed turn by turn",
  UNAVAILABLE: "Not supported by the supplied data or models",
};

export function CategoryPill({ c, conf }) {
  if (!c) return <span className="muted">N/A</span>;
  return (
    <span className="pill">
      <span className="sw" style={{ background: CAT_COLOR[c] || "#8494aa" }} />
      {CAT_LABEL[c] || c}
      {conf !== undefined && conf !== null && <span className="muted num">{fmtPct(conf, 0)}</span>}
    </span>
  );
}

export function Card({ title, sub, right, children, className = "", bodyClass = "card-b" }) {
  return (
    <section className={`card ${className}`}>
      {(title || right) && (
        <div className="card-h">
          <div>
            {title && <h3>{title}</h3>}
            {sub && <div className="card-sub">{sub}</div>}
          </div>
          {right}
        </div>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

export function Kpi({ label, value, status, note }) {
  return (
    <div className="card kpi">
      <div className="label">
        <span>{label}</span>
        {status && <StatusBadge s={status} />}
      </div>
      <div className="value num">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function Loading({ rows = 4, h = 18 }) {
  return (
    <div style={{ padding: 16, display: "grid", gap: 10 }} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skel" style={{ height: h, width: `${90 - i * 9}%` }} />
      ))}
    </div>
  );
}
export function ErrorState({ error, retry }) {
  return (
    <div className="state" role="alert">
      <h4>Unable to load data</h4>
      <div className="small">{error?.message || String(error)}</div>
      {retry && <button className="btn mt" onClick={retry}>Retry</button>}
    </div>
  );
}
export function Empty({ title = "No data available", children }) {
  return (
    <div className="state">
      <h4>{title}</h4>
      {children && <div className="small">{children}</div>}
    </div>
  );
}
export function Unavailable({ title, children }) {
  return (
    <div className="banner" style={{ alignItems: "center" }}>
      <StatusBadge s="UNAVAILABLE" />
      <div>
        <strong style={{ fontWeight: 600 }}>{title}</strong>
        {children && <div className="small muted">{children}</div>}
      </div>
    </div>
  );
}

// Wraps a useApi result into loading / error / empty / content
export function Async({ q, children, empty, rows }) {
  if (q.loading && !q.data) return <Loading rows={rows} />;
  if (q.error) return <ErrorState error={q.error} retry={q.retry} />;
  if (!q.data || (empty && empty(q.data))) return <Empty />;
  return children(q.data);
}

export function Meter({ value, color }) {
  return (
    <div className="meter" role="meter" aria-valuenow={Math.round((value || 0) * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${Math.max(0, Math.min(1, value || 0)) * 100}%`, background: color }} />
    </div>
  );
}

export function Radial({ value, size = 104, label }) {
  const r = (size - 12) / 2;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <svg width={size} height={size} role="img" aria-label={`${label}: ${fmtPct(value, 0)}`}>
      <circle cx={size / 2} cy={size / 2} r={r} stroke="#e6f2ff" strokeWidth="9" fill="none" />
      <circle cx={size / 2} cy={size / 2} r={r} stroke="#3b82f6" strokeWidth="9" fill="none" strokeLinecap="round"
        strokeDasharray={`${c * v} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} style={{ transition: "stroke-dasharray .4s ease" }} />
      <text x="50%" y="48%" textAnchor="middle" dominantBaseline="middle" fontSize="22" fontWeight="700" fill="#172033">
        {value === null || value === undefined ? "–" : Math.round(v * 100)}
        <tspan fontSize="12" fill="#5b6b82">%</tspan>
      </text>
      <text x="50%" y="68%" textAnchor="middle" fontSize="10" fill="#5b6b82">{label}</text>
    </svg>
  );
}

// Client-side sortable table (server-side paging/sorting is handled by pages that need it)
export function DataTable({ columns, rows, onRowClick, sort, onSort, selectedKey, rowKey = "id", empty }) {
  if (!rows?.length) return <Empty title="No matching records">{empty}</Empty>;
  return (
    <div className="table-wrap">
      <table className="t">
        <thead>
          <tr>
            {columns.map((c) => {
              const active = sort && sort.key === c.key;
              return (
                <th key={c.key} className={c.sortable ? "sortable" : ""} style={{ textAlign: c.align }}
                  onClick={() => c.sortable && onSort && onSort(c.key)}
                  aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : undefined}>
                  {c.label} {active ? (sort.dir === "asc" ? "▲" : "▼") : ""}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r[rowKey] ?? i} className={`${onRowClick ? "click" : ""} ${selectedKey === r[rowKey] ? "sel" : ""}`}
              onClick={() => onRowClick && onRowClick(r)} tabIndex={onRowClick ? 0 : undefined}
              onKeyDown={(e) => e.key === "Enter" && onRowClick && onRowClick(r)}>
              {columns.map((c) => (
                <td key={c.key} style={{ textAlign: c.align }} className={c.num ? "num" : ""}>
                  {c.render ? c.render(r) : r[c.key] ?? "N/A"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function useClientSort(rows, initial) {
  const [sort, setSort] = useState(initial);
  const sorted = useMemo(() => {
    if (!rows || !sort) return rows;
    const s = [...rows].sort((a, b) => {
      const x = a[sort.key], y = b[sort.key];
      if (x === y) return 0;
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return x > y ? 1 : -1;
    });
    return sort.dir === "asc" ? s : s.reverse();
  }, [rows, sort]);
  const onSort = (key) => setSort((s) => ({ key, dir: s && s.key === key && s.dir === "desc" ? "asc" : "desc" }));
  return { rows: sorted, sort, onSort };
}

export function Pager({ page, size, total, onPage }) {
  const pages = Math.max(1, Math.ceil(total / size));
  return (
    <div className="pager">
      <span className="num">{total ? `${(page - 1) * size + 1}–${Math.min(page * size, total)} of ${fmtInt(total)}` : "0 results"}</span>
      <div className="row">
        <button className="btn sm" disabled={page <= 1} onClick={() => onPage(page - 1)}>Previous</button>
        <span className="num">Page {page} / {pages}</span>
        <button className="btn sm" disabled={page >= pages} onClick={() => onPage(page + 1)}>Next</button>
      </div>
    </div>
  );
}

export function PageHead({ title, desc, right }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {desc && <p>{desc}</p>}
      </div>
      {right}
    </div>
  );
}
