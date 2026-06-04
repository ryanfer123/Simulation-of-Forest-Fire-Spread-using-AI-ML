"use client";

import { useState, useEffect, useCallback } from "react";
import DashboardHeader from "@/components/dashboard-header";
import FireMap, { type MapLayer } from "@/components/fire-map";
import MetricsPanel from "@/components/metrics-panel";
import MapLegend from "@/components/map-legend";
import { Download } from "lucide-react";

function ExportButton() {
  const [open, setOpen] = useState(false);
  const options = [
    { label: "Export as GeoTIFF", sub: "Fire risk raster · current view", ext: "geotiff" },
    { label: "Export as CSV",     sub: "Zone metrics · FIRMS format",      ext: "csv"     },
    { label: "Export Simulation", sub: "Spread timesteps · GeoJSON",       ext: "geojson" },
  ];
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1.5 px-2.5 py-1 rounded border text-[9.5px] font-mono font-semibold transition-colors"
        style={{ borderColor: "oklch(0.68 0.21 40)", color: "oklch(0.68 0.21 40)", background: "oklch(0.68 0.21 40)/8" }}
        aria-haspopup="true"
        aria-expanded={open}
      >
        <Download size={10} />
        Export GeoTIFF / CSV
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} aria-hidden="true" />
          <div className="absolute right-0 mt-1 w-52 rounded-lg border border-border bg-card shadow-xl z-20 overflow-hidden" role="menu">
            <p className="px-3 pt-2 pb-1 text-[8px] font-mono text-muted-foreground uppercase tracking-widest border-b border-border">
              NASA FIRMS MODIS/VIIRS
            </p>
            {options.map((o) => (
              <button
                key={o.ext}
                onClick={() => { alert(`Export triggered: pyrosense_export.${o.ext}`); setOpen(false); }}
                className="w-full text-left px-3 py-2 hover:bg-secondary transition-colors"
                role="menuitem"
              >
                <p className="text-[9.5px] font-mono text-foreground">{o.label}</p>
                <p className="text-[8px] font-mono text-muted-foreground">{o.sub}</p>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export default function Home() {
  const [activeLayers, setActiveLayers] = useState<Set<MapLayer>>(
    new Set(["risk", "historical"])
  );
  const [simulationActive, setSimulationActive] = useState(false);
  const [simulationStep, setSimulationStep] = useState(0);

  // Simulation tick
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
      if (next.has(layer)) {
        next.delete(layer);
      } else {
        next.add(layer);
      }
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
    // Ensure spread layer is on
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
    <div className="flex flex-col h-screen overflow-hidden bg-background text-foreground font-mono">
      <DashboardHeader simulationActive={simulationActive} />

      {/* Main layout */}
      <main className="flex flex-1 overflow-hidden">
        {/* Left: map controls */}
        <aside className="hidden lg:flex flex-col gap-3 w-52 shrink-0 p-3 overflow-y-auto border-r border-border">
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
        <section className="flex-1 flex flex-col p-3 gap-3 overflow-hidden">
          {/* Map top bar */}
          <div className="flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">
                Region View
              </span>
              <span className="text-[9px] font-mono text-foreground">Iberian Peninsula · Lat 40.4° N · Lon 3.7° W</span>
            </div>
            {/* Mobile layer toggle row */}
            <div className="flex items-center gap-1.5 lg:hidden">
              {(["risk", "historical", "spread"] as MapLayer[]).map((l) => (
                <button
                  key={l}
                  onClick={() => toggleLayer(l)}
                  className="text-[8.5px] font-mono px-2 py-0.5 rounded border transition-colors"
                  style={
                    activeLayers.has(l)
                      ? { borderColor: "oklch(0.68 0.21 40)", color: "oklch(0.68 0.21 40)", background: "oklch(0.68 0.21 40)/10" }
                      : { borderColor: "oklch(0.26 0.015 30)", color: "oklch(0.55 0.01 60)", background: "transparent" }
                  }
                >
                  {l}
                </button>
              ))}
            </div>
            <div className="hidden lg:flex items-center gap-3">
              <div className="flex items-center gap-1 text-[9px] font-mono text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-[oklch(0.50_0.12_145)]" />
                Safe &nbsp;
                <span className="w-2 h-2 rounded-full bg-[oklch(0.78_0.16_70)]" />
                Monitor &nbsp;
                <span className="w-2 h-2 rounded-full bg-[oklch(0.68_0.21_40)]" />
                High &nbsp;
                <span className="w-2 h-2 rounded-full bg-[oklch(0.58_0.24_27)]" />
                Critical
              </div>
              <ExportButton />
            </div>
          </div>

          {/* Map */}
          <div className="flex-1 overflow-hidden rounded-lg border border-border">
            <FireMap
              activeLayers={activeLayers}
              simulationActive={simulationActive}
              simulationStep={simulationStep}
            />
          </div>

          {/* Bottom stat strip */}
          <div className="shrink-0 grid grid-cols-4 gap-2">
            {[
              { label: "Zones Monitored", value: "8" },
              { label: "At-Risk Area", value: "13,870 ha" },
              { label: "Ignition Probability", value: "92.4%", danger: true },
              { label: "Response Teams", value: "24 active" },
            ].map((s) => (
              <div
                key={s.label}
                className="bg-card border border-border rounded-lg px-3 py-2 flex flex-col"
              >
                <span className="text-[8.5px] font-mono text-muted-foreground uppercase tracking-wider">
                  {s.label}
                </span>
                <span
                  className="text-sm font-mono font-bold mt-0.5"
                  style={{ color: s.danger ? "oklch(0.58 0.24 27)" : "oklch(0.93 0.01 60)" }}
                >
                  {s.value}
                </span>
              </div>
            ))}
          </div>
        </section>

        {/* Right: metrics */}
        <aside className="hidden xl:flex flex-col w-72 shrink-0 p-3 overflow-hidden border-l border-border">
          <div className="shrink-0 mb-2 flex items-center justify-between">
            <h2 className="text-[10px] font-mono text-muted-foreground uppercase tracking-widest">
              AI Metrics
            </h2>
            <span className="text-[8.5px] font-mono text-muted-foreground">FIRMS MODIS/VIIRS</span>
          </div>
          <MetricsPanel />
        </aside>
      </main>

      {/* Mobile bottom panel (shows metrics below map on small screens) */}
      <div className="xl:hidden border-t border-border bg-card px-3 py-2">
          <div className="flex gap-3 overflow-x-auto pb-1">
          {[
            { label: "Accuracy",  value: "98.0%",     color: "oklch(0.50 0.12 145)" },
            { label: "F1 Score",  value: "0.928",     color: "oklch(0.50 0.12 145)" },
            { label: "IoU",       value: "0.866",     color: "oklch(0.55 0.15 200)" },
            { label: "Precision", value: "94.1%",     color: "oklch(0.78 0.16 70)" },
            { label: "Recall",    value: "91.6%",     color: "oklch(0.78 0.16 70)" },
            { label: "AUC-ROC",   value: "0.971",     color: "oklch(0.50 0.12 145)" },
          ].map((m) => (
            <div key={m.label} className="shrink-0 flex flex-col items-center px-3 py-1 border border-border rounded">
              <span className="text-[8px] font-mono text-muted-foreground">{m.label}</span>
              <span className="text-sm font-mono font-bold" style={{ color: m.color }}>{m.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
