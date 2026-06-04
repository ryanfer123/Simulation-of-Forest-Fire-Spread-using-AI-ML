"use client";

import { useEffect, useState } from "react";

interface Props {
  fireCount: number;
  dateRange: [string, string] | null;
}

export default function DashboardHeader({ fireCount, dateRange }: Props) {
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
        <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden="true">
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
          <h1 className="mono" style={{ fontSize: 14, fontWeight: 700, color: "var(--fg)", letterSpacing: "-0.02em" }}>
            Forest Fire Prediction
          </h1>
          <p className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", letterSpacing: "0.08em", textTransform: "uppercase" }}>
            NASA FIRMS · VIIRS Satellite Data
          </p>
        </div>
      </div>

      {/* Center: data summary */}
      <div className="flex-center gap-3">
        {dateRange && (
          <span className="pill pill--muted">
            {dateRange[0]} — {dateRange[1]}
          </span>
        )}
        <span className="pill pill--accent">
          {fireCount.toLocaleString()} detections
        </span>
        <span className="pill pill--success">
          <span className="pulse-dot" style={{ color: "var(--success)" }} />
          Model Trained
        </span>
      </div>

      {/* Right: region and clock */}
      <div className="flex-center gap-4">
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Region
          </span>
          <span className="mono" style={{ fontSize: 11, color: "var(--fg)", fontWeight: 500 }}>
            Uttarakhand, India
          </span>
          <span className="mono" style={{ fontSize: 8, color: "var(--fg-dim)" }}>
            77.0–81.0°E · 28.5–31.5°N
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
            Local Time
          </span>
          <span className="mono" style={{ fontSize: 13, color: "var(--fg)", fontWeight: 600, fontVariantNumeric: "tabular-nums" }}>
            {time}
          </span>
        </div>
      </div>
    </header>
  );
}
