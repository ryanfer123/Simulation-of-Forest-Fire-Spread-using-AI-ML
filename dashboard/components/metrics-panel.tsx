"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";

interface ModelMetrics {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  auc_roc: number;
  confusion_matrix: {
    true_positive: number;
    false_positive: number;
    true_negative: number;
    false_negative: number;
  };
  train_samples: number;
  test_samples: number;
  train_positives: number;
  test_positives: number;
  feature_importances: Record<string, number>;
  region: string;
  bounds: number[];
  grid_size: number;
  date_range: [string, string];
  dataset: string;
}

const CHART_TOOLTIP_STYLE = {
  background: "#151918",
  border: "1px solid #2a322e",
  borderRadius: 8,
  fontSize: 10,
  fontFamily: "'JetBrains Mono', monospace",
  boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
};

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/metrics")
      .then((r) => {
        if (!r.ok) throw new Error("Metrics not found");
        return r.json();
      })
      .then(setMetrics)
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 24 }}>
        <p className="mono" style={{ fontSize: 11, color: "var(--fg-muted)" }}>
          No model metrics available.
        </p>
        <p className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", marginTop: 6 }}>
          Run <code>python train_fire_model.py</code> first.
        </p>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="card" style={{ textAlign: "center", padding: 24 }}>
        <p className="mono" style={{ fontSize: 11, color: "var(--fg-muted)" }}>
          Loading metrics...
        </p>
      </div>
    );
  }

  const cm = metrics.confusion_matrix;
  const featureData = Object.entries(metrics.feature_importances)
    .map(([name, value]) => ({ name: name.replace(/_/g, " "), value: +(value * 100).toFixed(1) }))
    .sort((a, b) => b.value - a.value);

  return (
    <aside
      style={{ display: "flex", flexDirection: "column", gap: 12, height: "100%", overflowY: "auto", paddingRight: 2 }}
      aria-label="AI metrics panel"
    >
      {/* Dataset badge */}
      <div className="card" style={{
        background: "var(--accent-dim)", borderColor: "var(--accent-border)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <p className="mono" style={{ fontSize: 10, fontWeight: 700, color: "var(--accent)" }}>
            {metrics.dataset}
          </p>
          <p className="mono" style={{ fontSize: 9, color: "var(--fg-muted)", marginTop: 2 }}>
            {metrics.region} · {metrics.date_range[0]} to {metrics.date_range[1]}
          </p>
        </div>
        <span className="mono" style={{
          fontSize: 8, padding: "3px 8px", borderRadius: "var(--radius-sm)",
          background: "var(--info-dim)", border: "1px solid var(--info-border)",
          color: "var(--info)", fontWeight: 700,
        }}>
          {metrics.model.replace("Classifier", "")}
        </span>
      </div>

      {/* Model Performance */}
      <div>
        <p className="label" style={{ marginBottom: 8 }}>Classification Metrics</p>
        <div className="grid-2">
          {[
            { label: "Accuracy", value: `${(metrics.accuracy * 100).toFixed(2)}%`, color: "var(--success)" },
            { label: "F1 Score", value: metrics.f1_score.toFixed(4), color: metrics.f1_score > 0.5 ? "var(--success)" : "var(--warning)" },
            { label: "Precision", value: metrics.precision.toFixed(4), color: "var(--info)" },
            { label: "Recall", value: metrics.recall.toFixed(4), color: "var(--accent)" },
          ].map((m) => (
            <div key={m.label} className="stat-card">
              <span className="stat-label">{m.label}</span>
              <span className="stat-value" style={{ color: m.color, fontSize: 18 }}>{m.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* AUC-ROC */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="label">AUC-ROC</span>
          <span className="mono" style={{ fontSize: 16, fontWeight: 700, color: metrics.auc_roc > 0.7 ? "var(--success)" : "var(--warning)" }}>
            {metrics.auc_roc.toFixed(4)}
          </span>
        </div>
        <div className="progress-track" style={{ marginTop: 8 }}>
          <div className="progress-fill" style={{
            width: `${metrics.auc_roc * 100}%`,
            background: metrics.auc_roc > 0.7 ? "var(--success)" : "var(--warning)",
          }} />
        </div>
        <p className="mono" style={{ fontSize: 8, color: "var(--fg-dim)", marginTop: 6 }}>
          {metrics.auc_roc < 0.5 ? "Below random baseline — insufficient training data" :
           metrics.auc_roc < 0.7 ? "Weak discriminative power — more data needed" :
           "Good discriminative performance"}
        </p>
      </div>

      {/* Confusion Matrix */}
      <div className="card">
        <p className="label" style={{ marginBottom: 8 }}>Confusion Matrix</p>
        <div className="grid-2" style={{ marginBottom: 10 }}>
          {[
            { label: "True Pos", value: cm.true_positive, color: "var(--success)" },
            { label: "False Pos", value: cm.false_positive, color: "var(--accent)" },
            { label: "False Neg", value: cm.false_negative, color: "var(--danger)" },
            { label: "True Neg", value: cm.true_negative, color: "var(--success)" },
          ].map((c) => (
            <div key={c.label} className="confusion-cell"
              style={{ background: `color-mix(in srgb, ${c.color} 12%, transparent)` }}
            >
              <span className="confusion-cell-label">{c.label}</span>
              <span className="confusion-cell-value" style={{ color: c.color }}>{c.value.toLocaleString()}</span>
            </div>
          ))}
        </div>
        <div style={{
          display: "grid", gridTemplateColumns: "1fr 1fr",
          gap: "4px 16px", borderTop: "1px solid var(--border)", paddingTop: 8,
        }}>
          <div className="mono" style={{ fontSize: 9, color: "var(--fg-muted)" }}>
            Train samples: <span style={{ color: "var(--fg)", fontWeight: 600 }}>{metrics.train_samples.toLocaleString()}</span>
          </div>
          <div className="mono" style={{ fontSize: 9, color: "var(--fg-muted)" }}>
            Test samples: <span style={{ color: "var(--fg)", fontWeight: 600 }}>{metrics.test_samples.toLocaleString()}</span>
          </div>
          <div className="mono" style={{ fontSize: 9, color: "var(--fg-muted)" }}>
            Train positives: <span style={{ color: "var(--fg)", fontWeight: 600 }}>{metrics.train_positives}</span>
          </div>
          <div className="mono" style={{ fontSize: 9, color: "var(--fg-muted)" }}>
            Test positives: <span style={{ color: "var(--fg)", fontWeight: 600 }}>{metrics.test_positives}</span>
          </div>
        </div>
      </div>

      {/* Feature Importance */}
      <div className="card">
        <p className="label" style={{ marginBottom: 8 }}>Feature Importance (%)</p>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={featureData} layout="vertical" margin={{ top: 0, right: 4, left: 0, bottom: 0 }}>
            <XAxis type="number" tick={{ fontSize: 8, fill: "#7a8580" }} tickFormatter={(v: number) => `${v}%`} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 8, fill: "#7a8580" }} width={100} />
            <Tooltip contentStyle={CHART_TOOLTIP_STYLE} formatter={(v: number) => [`${v.toFixed(1)}%`, "Importance"]} />
            <Bar dataKey="value" fill="var(--accent)" radius={[0, 3, 3, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Data note */}
      <div className="card" style={{ borderColor: "var(--warning-border)" }}>
        <p className="label" style={{ marginBottom: 6, color: "var(--warning)" }}>Data Note</p>
        <p className="mono" style={{ fontSize: 9, color: "var(--fg-muted)", lineHeight: 1.6 }}>
          The FIRMS 7-day export covers {metrics.date_range[0]} to {metrics.date_range[1]} with
          only {metrics.train_positives + metrics.test_positives} fire-positive cells out of {(metrics.train_samples + metrics.test_samples).toLocaleString()} total.
          This extreme class imbalance ({((metrics.train_positives + metrics.test_positives) / (metrics.train_samples + metrics.test_samples) * 100).toFixed(2)}% positive rate)
          is typical of satellite fire detection data and explains the low recall.
          Production systems address this with longer temporal windows, oversampling, and additional feature sources (weather, terrain, NDVI).
        </p>
      </div>
    </aside>
  );
}
