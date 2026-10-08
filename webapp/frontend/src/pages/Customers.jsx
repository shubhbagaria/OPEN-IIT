import { useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { customersApi } from "../services/api";
import { useApi } from "../hooks/useApi";
import { Async, Card, CategoryPill, DataTable, PageHead, Pager, StatusBadge, fmtInr, fmtPct } from "../components/ui";

export default function Customers() {
  const nav = useNavigate();
  const [sp, setSp] = useSearchParams();
  const get = (k, d = "") => sp.get(k) ?? d;
  const [qText, setQText] = useState(get("q"));
  const params = { q: get("q"), portfolio: get("portfolio"), lender: get("lender"), bucket: get("bucket"),
    sort: get("sort", "n_ptps"), order: get("order", "desc"), page: Number(get("page", "1")), size: 25 };
  const set = (patch) => {
    const n = new URLSearchParams(sp);
    Object.entries(patch).forEach(([k, v]) => (v === "" ? n.delete(k) : n.set(k, v)));
    if (!("page" in patch)) n.delete("page");
    setSp(n);
  };
  useEffect(() => {
    const t = setTimeout(() => qText !== params.q && set({ q: qText }), 350);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qText]);
  const q = useApi((o) => customersApi.list(params, o), [sp.toString()]);
  const onSort = (key) => set({ sort: key, order: params.sort === key && params.order === "desc" ? "asc" : "desc" });
  const cols = [
    { key: "account_id", label: "Account", sortable: true, render: (r) => <span style={{ fontWeight: 600 }}>{r.account_id}</span> },
    { key: "lender_id", label: "Lender", sortable: true },
    { key: "portfolio", label: "Portfolio", sortable: true },
    { key: "bucket_start", label: "Bucket", sortable: true },
    { key: "dpd_start", label: "DPD", sortable: true, num: true },
    { key: "outstanding", label: "Outstanding", sortable: true, num: true, render: (r) => fmtInr(r.outstanding) },
    { key: "bureau_score_band", label: "Bureau band", sortable: true },
    { key: "n_ptps", label: "PTPs", sortable: true, num: true },
    { key: "keep_rate", label: "Keep rate", sortable: true, num: true, render: (r) => fmtPct(r.keep_rate, 0) },
    { key: "mean_confidence", label: "Mean credibility", sortable: true, num: true, render: (r) => fmtPct(r.mean_confidence, 0) },
    { key: "latest_category", label: "Latest category", render: (r) => <CategoryPill c={r.latest_category} /> },
  ];
  const f = q.data?.filters;
  return (
    <>
      <PageHead title="Customer Intelligence" desc="Accounts from accounts.csv with their PTP history. Names and contact details are not in the supplied data and are never shown." />
      <Card bodyClass="" title="Accounts" right={<div className="row" style={{ gap: 4 }}><StatusBadge s="ACTUAL" /><StatusBadge s="STORED MODEL OUTPUT" /></div>}>
        <div className="toolbar" style={{ padding: "12px 16px 0" }}>
          <input className="input" placeholder="Search account ID" value={qText} onChange={(e) => setQText(e.target.value)} aria-label="Search accounts" />
          <select className="select" value={params.portfolio} onChange={(e) => set({ portfolio: e.target.value })} aria-label="Portfolio">
            <option value="">All portfolios</option>{f?.portfolio.map((x) => <option key={x}>{x}</option>)}
          </select>
          <select className="select" value={params.lender} onChange={(e) => set({ lender: e.target.value })} aria-label="Lender">
            <option value="">All lenders</option>{f?.lender.map((x) => <option key={x}>{x}</option>)}
          </select>
          <select className="select" value={params.bucket} onChange={(e) => set({ bucket: e.target.value })} aria-label="Bucket">
            <option value="">All buckets</option>{f?.bucket.map((x) => <option key={x}>{x}</option>)}
          </select>
          <button className="btn" onClick={() => { setQText(""); setSp(new URLSearchParams()); }}>Clear filters</button>
        </div>
        <div style={{ marginTop: 12 }}>
          <Async q={q} rows={8}>
            {(d) => (
              <>
                <DataTable columns={cols} rows={d.rows} rowKey="account_id" sort={{ key: params.sort, dir: params.order }} onSort={onSort}
                  onRowClick={(r) => nav(`/customers/${r.account_id}`)} />
                <Pager page={d.page} size={d.size} total={d.total} onPage={(p) => set({ page: p })} />
              </>
            )}
          </Async>
        </div>
      </Card>
    </>
  );
}
