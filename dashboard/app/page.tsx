"use client";

import { useState, useEffect, useCallback } from "react";
import DashboardHeader from "@/components/dashboard-header";
import FireMap, { type MapLayer } from "@/components/fire-map";
import MetricsPanel from "@/components/metrics-panel";
import MapLegend from "@/components/map-legend";

export default function Home() {
  const [activeLayers, setActiveLayers] = useState<Set<MapLayer>>(
    new Set(["risk", "historical"])
  );
  const [simulationActive, setSimulationActive] = useState(false);
  const [simulationStep, setSimulationStep] = useState(0);

  useEffect(() => {
    if (!simulationActive) return;
    const id = setInterval(() => {
      setSimulationStep((s) => {
        if (s >= 12) {
          setSimulationActive(false);
          return 12;
        }
        return s + 1;
      });
    }, 900);
    return () => clearInterval(id);
  }, [simulationActive]);

  const toggleLayer = useCallback((layer: MapLayer) => {
    setActiveLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) next.delete(layer);
      else next.add(layer);
      return next;
    });
  }, []);

  const toggleSimulation = useCallback(() => {
    if (simulationStep >= 12) {
      setSimulationStep(0);
      setSimulationActive(true);
    } else {
      setSimulationActive((v) => !v);
    }
    setActiveLayers((prev) => {
      const next = new Set(prev);
      next.add("spread");
      return next;
    });
  }, [simulationStep]);

  const resetSimulation = useCallback(() => {
    setSimulationActive(false);
    setSimulationStep(0);
  }, []);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      overflow: "hidden",
      background: "var(--bg)",
      color: "var(--fg)",
      fontFamily: "var(--font-mono)",
    }}>
      <DashboardHeader simulationActive={simulationActive} />

      {/* Main layout */}
      <main style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Left sidebar: map controls */}
        <aside style={{
          display: "flex",
          flexDirection: "column",
          gap: 12,
          width: 220,
          flexShrink: 0,
          padding: 12,
          overflowY: "auto",
          borderRight: "1px solid var(--border)",
        }}>
          <MapLegend
            activeLayers={activeLayers}
            onToggle={toggleLayer}
            simulationActive={simulationActive}
            simulationStep={simulationStep}
            onToggleSimulation={toggleSimulation}
            onResetSimulation={resetSimulation}
          />
        </aside>

        {/* Center: map */}
        <section style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: 12,
          gap: 12,
          overflow: "hidden",
        }}>
          {/* Map top bar */}
          <div className="flex-between" style={{ flexShrink: 0 }}>
            <div className="flex-center gap-2">
              <span className="label">Region View</span>
              <span className="mono" style={{ fontSize: 10, color: "var(--fg)" }}>
                Iberian Peninsula · Lat 40.4° N · Lon 3.7° W
              </span>
            </div>
            <div className="flex-center gap-3">
              <div className="flex-center gap-2" style={{ fontSize: 9 }}>
                {[
                  { label: "Safe", color: "var(--success)" },
                  { label: "Monitor", color: "var(--warning)" },
                  { label: "High", color: "var(--accent)" },
                  { label: "Critical", color: "var(--danger)" },
                ].map((r) => (
                  <span key={r.label} className="flex-center gap-1 mono" style={{ color: "var(--fg-muted)", display: "inline-flex", alignItems: "center", gap: 4 }}>
                    <span className="risk-dot" style={{ background: r.color, width: 8, height: 8 }} />
                    {r.label}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Map */}
          <div style={{
            flex: 1,
            overflow: "hidden",
            borderRadius: "var(--radius)",
            border: "1px solid var(--border)",
          }}>
            <FireMap
              activeLayers={activeLayers}
              simulationActive={simulationActive}
              simulationStep={simulationStep}
            />
          </div>

          {/* Bottom stat strip */}
          <div className="grid-4" style={{ flexShrink: 0 }}>
            {[
              { label: "Zones Monitored", value: "8", color: "var(--fg)" },
              { label: "At-Risk Area", value: "13,870 ha", color: "var(--accent)" },
              { label: "Ignition Probability", value: "92.4%", color: "var(--danger)" },
              { label: "Response Teams", value: "24 active", color: "var(--success)" },
            ].map((s) => (
              <div key={s.label} className="stat-card">
                <span className="stat-label">{s.label}</span>
                <span className="stat-value" style={{ color: s.color, fontSize: 16 }}>{s.value}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Right sidebar: metrics */}
        <aside style={{
          display: "flex",
          flexDirection: "column",
          width: 300,
          flexShrink: 0,
          padding: 12,
          overflow: "hidden",
          borderLeft: "1px solid var(--border)",
        }}>
          <div className="flex-between" style={{ marginBottom: 10, flexShrink: 0 }}>
            <h2 className="label">AI Metrics</h2>
            <span className="mono" style={{ fontSize: 8.5, color: "var(--fg-dim)" }}>FIRMS MODIS/VIIRS</span>
          </div>
          <MetricsPanel />
        </aside>
      </main>
    </div>
  );
}
