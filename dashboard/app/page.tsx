"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import DashboardHeader from "@/components/dashboard-header";
import MetricsPanel from "@/components/metrics-panel";

// Leaflet uses window/document, so we must load it client-side only
const FireMap = dynamic(() => import("@/components/fire-map"), { ssr: false });

export default function Home() {
  const [fireCount, setFireCount] = useState(0);
  const [dateRange, setDateRange] = useState<[string, string] | null>(null);

  useEffect(() => {
    fetch("/api/fires")
      .then((r) => r.json())
      .then((data) => {
        setFireCount(data.total ?? 0);
        if (data.points && data.points.length > 0) {
          const dates = [...new Set(data.points.map((p: any) => p.acq_date))] as string[];
          dates.sort();
          setDateRange([dates[0], dates[dates.length - 1]]);
        }
      })
      .catch(() => {});
  }, []);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      overflow: "hidden",
      background: "var(--bg)",
      color: "var(--fg)",
    }}>
      <DashboardHeader fireCount={fireCount} dateRange={dateRange} />

      <main style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Map */}
        <section style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: 12,
          gap: 12,
          overflow: "hidden",
        }}>
          <div className="flex-between" style={{ flexShrink: 0 }}>
            <span className="label">Satellite Fire Detections — Uttarakhand, India</span>
            <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)" }}>
              Source: NASA FIRMS VIIRS (SUOMI NPP)
            </span>
          </div>
          <div style={{
            flex: 1,
            overflow: "hidden",
            borderRadius: "var(--radius)",
            border: "1px solid var(--border)",
          }}>
            <FireMap />
          </div>
        </section>

        {/* Right sidebar: metrics */}
        <aside style={{
          display: "flex",
          flexDirection: "column",
          width: 320,
          flexShrink: 0,
          padding: 12,
          overflow: "hidden",
          borderLeft: "1px solid var(--border)",
        }}>
          <div className="flex-between" style={{ marginBottom: 10, flexShrink: 0 }}>
            <h2 className="label">Model Evaluation</h2>
            <span className="mono" style={{ fontSize: 8, color: "var(--fg-dim)" }}>
              GradientBoosting
            </span>
          </div>
          <MetricsPanel />
        </aside>
      </main>
    </div>
  );
}
