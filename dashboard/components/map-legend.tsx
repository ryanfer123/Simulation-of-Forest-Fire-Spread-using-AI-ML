"use client";

import type { MapLayer } from "./fire-map";

interface Props {
  activeLayers: Set<MapLayer>;
  onToggle: (layer: MapLayer) => void;
  simulationActive: boolean;
  simulationStep: number;
  onToggleSimulation: () => void;
  onResetSimulation: () => void;
}

const LAYERS: { id: MapLayer; label: string; color: string }[] = [
  { id: "risk", label: "Risk Zones", color: "var(--accent)" },
  { id: "historical", label: "Historical Fires", color: "#a0522d" },
  { id: "spread", label: "Spread Simulation", color: "var(--danger)" },
];

const RISK_LEGEND = [
  { label: "Critical", color: "var(--danger)" },
  { label: "High", color: "var(--accent)" },
  { label: "Medium", color: "var(--warning)" },
  { label: "Low", color: "var(--success)" },
];

export default function MapLegend({
  activeLayers,
  onToggle,
  simulationActive,
  simulationStep,
  onToggleSimulation,
  onResetSimulation,
}: Props) {
  return (
    <div className="flex-col gap-3" style={{ display: "flex" }}>
      {/* Layer toggles */}
      <div className="card">
        <p className="label" style={{ marginBottom: 10 }}>Map Layers</p>
        <div className="flex-col gap-1" style={{ display: "flex" }}>
          {LAYERS.map((l) => {
            const active = activeLayers.has(l.id);
            return (
              <button
                key={l.id}
                onClick={() => onToggle(l.id)}
                className="layer-toggle"
                aria-pressed={active}
              >
                <span
                  className="layer-checkbox"
                  style={{
                    background: active ? l.color : "transparent",
                    borderColor: l.color,
                    opacity: active ? 1 : 0.5,
                  }}
                >
                  {active && (
                    <svg viewBox="0 0 12 12" fill="none">
                      <path d="M2.5 6L5 8.5L9.5 3.5" stroke="var(--bg)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  )}
                </span>
                <span
                  className="layer-label"
                  style={{ color: active ? "var(--fg)" : "var(--fg-dim)" }}
                >
                  {l.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Risk legend */}
      <div className="card">
        <p className="label" style={{ marginBottom: 10 }}>Risk Level</p>
        <div className="flex-col gap-2" style={{ display: "flex" }}>
          {RISK_LEGEND.map((r) => (
            <div key={r.label} className="flex-center gap-2">
              <span className="risk-dot" style={{ background: r.color }} />
              <span className="mono" style={{ fontSize: 10.5, color: "var(--fg-muted)" }}>{r.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Simulation control */}
      <div className="card">
        <p className="label" style={{ marginBottom: 10 }}>Simulation</p>
        <div className="flex-col gap-2" style={{ display: "flex" }}>
          <div className="flex-between">
            <span className="mono" style={{ fontSize: 10, color: "var(--fg-muted)" }}>Step</span>
            <span className="mono" style={{ fontSize: 10, color: "var(--accent)", fontWeight: 700 }}>
              T+{simulationStep * 6}h
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min((simulationStep / 12) * 100, 100)}%`,
                background: "var(--danger)",
              }}
            />
          </div>
          <div className="flex-center gap-2" style={{ marginTop: 4 }}>
            <button
              onClick={onToggleSimulation}
              className={simulationActive ? "btn btn--danger" : "btn btn--accent"}
              style={{ flex: 1 }}
            >
              {simulationActive ? "■ Pause" : "▶ Run"}
            </button>
            <button onClick={onResetSimulation} className="btn btn--ghost">
              ↺
            </button>
          </div>
        </div>
      </div>

      {/* Environmental conditions */}
      <div className="card">
        <p className="label" style={{ marginBottom: 10 }}>Conditions</p>
        <div className="flex-col gap-1" style={{ display: "flex" }}>
          {[
            { k: "Temperature", v: "38°C", color: "var(--danger)" },
            { k: "Humidity", v: "12%", color: "var(--warning)" },
            { k: "Wind Speed", v: "18 km/h", color: "var(--accent)" },
            { k: "Fuel Moisture", v: "8%", color: "var(--danger)" },
            { k: "NDVI", v: "0.31", color: "var(--success)" },
          ].map(({ k, v, color }) => (
            <div key={k} className="condition-row">
              <span className="condition-key">{k}</span>
              <span className="condition-val" style={{ color }}>{v}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
