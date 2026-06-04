"use client";

import { useState, useEffect } from "react";

interface Props {
  simulationActive: boolean;
}

export default function DashboardHeader({ simulationActive }: Props) {
  const [time, setTime] = useState<string>("");

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
    <header style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "10px 20px",
      borderBottom: "1px solid var(--border)",
      background: "var(--bg-card)",
      flexShrink: 0,
    }}>
      {/* Brand */}
      <div className="flex-center gap-3">
        <svg width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <rect width="32" height="32" rx="8" fill="var(--danger)" opacity="0.15" />
          <path
            d="M16 5C16 5 11.5 10.5 11.5 15C11.5 17.5 13 19.5 15 20C14.4 18.2 15 15.8 16 14.5C17 16 17.6 17.5 17 20C18.8 19.5 20.5 17.5 20.5 15C20.5 10.5 16 5 16 5Z"
            fill="var(--accent)"
          />
          <path
            d="M16 20.5C16 20.5 13.5 22.5 13.5 24.5C13.5 25.7 14.6 27 16 27C17.4 27 18.5 25.7 18.5 24.5C18.5 22.5 16 20.5 16 20.5Z"
            fill="var(--danger)"
          />
        </svg>
        <div>
          <h1 className="mono" style={{ fontSize: 15, fontWeight: 700, color: "var(--fg)", letterSpacing: "-0.02em" }}>
            PyroSense
          </h1>
          <p className="mono" style={{ fontSize: 9.5, color: "var(--fg-dim)", letterSpacing: "0.12em", textTransform: "uppercase" }}>
            AI Forest Fire Prediction
          </p>
        </div>
      </div>

      {/* Center status pills */}
      <div className="flex-center gap-2" style={{ display: "flex" }}>
        <span className="pill pill--danger">
          <span className="pulse-dot" style={{ color: "var(--danger)" }} />
          ALERT LEVEL <strong>RED</strong>
        </span>
        <span className="pill pill--accent">
          ACTIVE ZONES <strong>8</strong>
        </span>
        <span className={simulationActive ? "pill pill--success" : "pill pill--muted"}>
          {simulationActive && <span className="pulse-dot" style={{ color: "var(--success)" }} />}
          SIMULATION <strong>{simulationActive ? "RUNNING" : "PAUSED"}</strong>
        </span>
        <span className="pill pill--success">
          MODEL <strong>ONLINE</strong>
        </span>
      </div>

      {/* Right side */}
      <div className="flex-center gap-4">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", letterSpacing: "0.12em", textTransform: "uppercase" }}>Region</span>
          <span className="mono" style={{ fontSize: 12, color: "var(--fg)", fontWeight: 500 }}>Uttarakhand, IN</span>
          <span className="mono" style={{ fontSize: 8, color: "var(--fg-dim)", fontVariantNumeric: "tabular-nums" }}>
            77.0–81.5°E &nbsp; 28.5–31.5°N
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", letterSpacing: "0.12em", textTransform: "uppercase" }}>UTC</span>
          <span className="mono" style={{ fontSize: 13, color: "var(--fg)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>{time}</span>
        </div>
        <span className="pill pill--danger" style={{ fontSize: 11 }}>
          <span className="pulse-dot" style={{ color: "var(--danger)" }} />
          LIVE
        </span>
      </div>
    </header>
  );
}
