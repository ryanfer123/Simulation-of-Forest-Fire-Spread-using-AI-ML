"use client";

import { useState, useEffect } from "react";

interface Props {
  simulationActive: boolean;
}

export default function DashboardHeader({ simulationActive }: Props) {
  const [time, setTime] = useState<string>("");
  const [modelStatus, setModelStatus] = useState("ONLINE");

  useEffect(() => {
    const update = () =>
      setTime(
        new Date().toLocaleTimeString("en-US", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
          hour12: false,
        })
      );
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-card shrink-0">
      {/* Brand */}
      <div className="flex items-center gap-3">
        {/* Fire icon mark */}
        <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
          <rect width="28" height="28" rx="7" fill="oklch(0.58 0.24 27)" opacity="0.15" />
          <path
            d="M14 4C14 4 10 9 10 13C10 15.2 11.4 17 13 17.5C12.5 16 13 14 14 13C15 15 15.5 16 15 17.5C16.6 17 18 15.2 18 13C18 9 14 4 14 4Z"
            fill="oklch(0.68 0.21 40)"
          />
          <path
            d="M14 18C14 18 11 20 11 22C11 23.1 12.3 24 14 24C15.7 24 17 23.1 17 22C17 20 14 18 14 18Z"
            fill="oklch(0.58 0.24 27)"
          />
        </svg>
        <div>
          <h1 className="text-sm font-mono font-bold text-foreground tracking-tight">PyroSense</h1>
          <p className="text-[9px] font-mono text-muted-foreground tracking-widest uppercase">
            AI Forest Fire Prediction
          </p>
        </div>
      </div>

      {/* Center status pills */}
      <div className="hidden md:flex items-center gap-2">
        <StatusPill
          label="ALERT LEVEL"
          value="RED"
          color="oklch(0.58 0.24 27)"
          pulse
        />
        <StatusPill
          label="ACTIVE ZONES"
          value="8"
          color="oklch(0.68 0.21 40)"
        />
        <StatusPill
          label="SIMULATION"
          value={simulationActive ? "RUNNING" : "PAUSED"}
          color={simulationActive ? "oklch(0.50 0.12 145)" : "oklch(0.55 0.01 60)"}
          pulse={simulationActive}
        />
        <StatusPill
          label="MODEL"
          value={modelStatus}
          color="oklch(0.50 0.12 145)"
        />
      </div>

      {/* Right: time & region */}
      <div className="flex items-center gap-4">
        <div className="hidden lg:flex flex-col items-end">
          <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">Region</span>
          <span className="text-xs font-mono text-foreground">Uttarakhand, IN</span>
          <span className="text-[8px] font-mono text-muted-foreground tabular-nums">
            77.0&ndash;81.5&deg;E &nbsp; 28.5&ndash;31.5&deg;N
          </span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">UTC</span>
          <span className="text-xs font-mono text-foreground tabular-nums">{time}</span>
        </div>
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded border text-[10px] font-mono font-semibold"
          style={{
            background: "oklch(0.58 0.24 27)/10",
            borderColor: "oklch(0.58 0.24 27)/40",
            color: "oklch(0.58 0.24 27)",
          }}
        >
          <span className="relative flex h-1.5 w-1.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[oklch(0.58_0.24_27)] opacity-60" />
            <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-[oklch(0.58_0.24_27)]" />
          </span>
          LIVE
        </div>
      </div>
    </header>
  );
}

function StatusPill({
  label,
  value,
  color,
  pulse = false,
}: {
  label: string;
  value: string;
  color: string;
  pulse?: boolean;
}) {
  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded border"
      style={{
        background: `${color}12`,
        borderColor: `${color}35`,
      }}
    >
      {pulse && (
        <span className="relative flex h-1.5 w-1.5">
          <span
            className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-60"
            style={{ background: color }}
          />
          <span
            className="relative inline-flex rounded-full h-1.5 w-1.5"
            style={{ background: color }}
          />
        </span>
      )}
      <span className="text-[8.5px] font-mono text-muted-foreground uppercase tracking-wider">{label}</span>
      <span className="text-[10px] font-mono font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}
