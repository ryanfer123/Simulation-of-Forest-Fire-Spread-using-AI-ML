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
  { id: "risk", label: "Risk Zones", color: "oklch(0.68 0.21 40)" },
  { id: "historical", label: "Historical Fires", color: "oklch(0.45 0.10 30)" },
  { id: "spread", label: "Spread Simulation", color: "oklch(0.58 0.24 27)" },
];

const RISK_LEGEND = [
  { label: "Critical", color: "oklch(0.58 0.24 27)" },
  { label: "High", color: "oklch(0.68 0.21 40)" },
  { label: "Medium", color: "oklch(0.78 0.16 70)" },
  { label: "Low", color: "oklch(0.50 0.12 145)" },
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
    <div className="flex flex-col gap-3">
      {/* Layer toggles */}
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Map Layers</p>
        <div className="flex flex-col gap-1.5">
          {LAYERS.map((l) => {
            const active = activeLayers.has(l.id);
            return (
              <button
                key={l.id}
                onClick={() => onToggle(l.id)}
                className="flex items-center gap-2 text-left w-full group"
                aria-pressed={active}
              >
                <span
                  className="w-3 h-3 rounded-sm border transition-all duration-150 shrink-0"
                  style={{
                    background: active ? l.color : "transparent",
                    borderColor: l.color,
                    opacity: active ? 1 : 0.5,
                  }}
                />
                <span
                  className="text-[10px] font-mono transition-colors duration-150"
                  style={{ color: active ? "oklch(0.93 0.01 60)" : "oklch(0.55 0.01 60)" }}
                >
                  {l.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Risk legend */}
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Risk Level</p>
        <div className="flex flex-col gap-1.5">
          {RISK_LEGEND.map((r) => (
            <div key={r.label} className="flex items-center gap-2">
              <span
                className="w-3 h-3 rounded-full shrink-0"
                style={{ background: r.color }}
              />
              <span className="text-[10px] font-mono text-muted-foreground">{r.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Simulation control */}
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Simulation</p>
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono text-muted-foreground">Step</span>
            <span className="text-[10px] font-mono" style={{ color: "oklch(0.68 0.21 40)" }}>
              T+{simulationStep * 6}h
            </span>
          </div>
          {/* Progress bar */}
          <div className="h-1.5 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.min((simulationStep / 12) * 100, 100)}%`,
                background: "oklch(0.58 0.24 27)",
              }}
            />
          </div>
          <div className="flex gap-1.5 mt-1">
            <button
              onClick={onToggleSimulation}
              className="flex-1 text-[10px] font-mono font-semibold py-1.5 rounded border transition-all duration-150"
              style={
                simulationActive
                  ? { background: "oklch(0.58 0.24 27)/15", borderColor: "oklch(0.58 0.24 27)/60", color: "oklch(0.58 0.24 27)" }
                  : { background: "oklch(0.68 0.21 40)/15", borderColor: "oklch(0.68 0.21 40)/60", color: "oklch(0.68 0.21 40)" }
              }
            >
              {simulationActive ? "■ Pause" : "▶ Run"}
            </button>
            <button
              onClick={onResetSimulation}
              className="px-2 text-[10px] font-mono py-1.5 rounded border border-border text-muted-foreground hover:text-foreground transition-colors"
            >
              ↺
            </button>
          </div>
        </div>
      </div>

      {/* Environmental conditions */}
      <div className="bg-card border border-border rounded-lg p-3">
        <p className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-2">Conditions</p>
        <div className="flex flex-col gap-1.5">
          {[
            { k: "Temperature", v: "38°C", w: 0.88 },
            { k: "Humidity", v: "12%", w: 0.12 },
            { k: "Wind Speed", v: "18 km/h", w: 0.6 },
            { k: "Fuel Moisture", v: "8%", w: 0.08 },
            { k: "NDVI", v: "0.31", w: 0.31 },
          ].map(({ k, v, w }) => (
            <div key={k} className="flex items-center justify-between">
              <span className="text-[9.5px] font-mono text-muted-foreground">{k}</span>
              <span
                className="text-[9.5px] font-mono font-semibold"
                style={{ color: `oklch(${0.45 + w * 0.35} ${0.08 + w * 0.17} ${145 - w * 120})` }}
              >
                {v}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
