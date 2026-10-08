import { lazy, Suspense } from "react";
import { Navigate, Route, Routes, Link } from "react-router-dom";
import Layout from "./components/Layout";
import { HealthProvider } from "./components/Health";
const Overview = lazy(() => import("./pages/Overview"));
const LiveCalls = lazy(() => import("./pages/LiveCalls"));
const LiveCallDetail = lazy(() => import("./pages/LiveCallDetail"));
const Customers = lazy(() => import("./pages/Customers"));
const CustomerDetail = lazy(() => import("./pages/CustomerDetail"));
const Agents = lazy(() => import("./pages/Agents"));
const AgentDetail = lazy(() => import("./pages/AgentDetail"));
const Treatment = lazy(() => import("./pages/Treatment"));
const Analytics = lazy(() => import("./pages/Analytics"));
const Audit = lazy(() => import("./pages/Audit"));
const AuditDetail = lazy(() => import("./pages/AuditDetail"));
import { Empty, Loading } from "./components/ui";

export default function App() {
  return (
    <HealthProvider>
      <Layout>
        <Suspense fallback={<Loading rows={6} />}>
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/live-calls" element={<LiveCalls />} />
          <Route path="/live-calls/:callId" element={<LiveCallDetail />} />
          <Route path="/customers" element={<Customers />} />
          <Route path="/customers/:customerId" element={<CustomerDetail />} />
          <Route path="/agents" element={<Agents />} />
          <Route path="/agents/:agentId" element={<AgentDetail />} />
          <Route path="/treatment" element={<Treatment />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/audit" element={<Audit />} />
          <Route path="/audit/:auditId" element={<AuditDetail />} />
          <Route path="*" element={<Empty title="Page not found"><Link to="/overview">Go to Overview</Link></Empty>} />
        </Routes>
        </Suspense>
      </Layout>
    </HealthProvider>
  );
}
