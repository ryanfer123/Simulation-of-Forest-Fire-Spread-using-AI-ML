"use client";

import { useEffect, useState, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface FirePoint {
  lat: number;
  lon: number;
  bright_ti4: number;
  frp: number;
  confidence: string;
  acq_date: string;
  daynight: string;
}

interface FireData {
  total: number;
  bounds: { minLon: number; minLat: number; maxLon: number; maxLat: number };
  points: FirePoint[];
}

function frpToColor(frp: number): string {
  if (frp > 15) return "#ef4444";
  if (frp > 5) return "#f97316";
  if (frp > 1) return "#eab308";
  return "#22c55e";
}

function frpToRadius(frp: number): number {
  return Math.max(3, Math.min(10, 3 + frp * 0.5));
}

export default function FireMap() {
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMap = useRef<L.Map | null>(null);
  const [fireData, setFireData] = useState<FireData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState<string>("all");

  // Fetch fire data from our API
  useEffect(() => {
    fetch("/api/fires")
      .then((r) => r.json())
      .then((data: FireData) => {
        setFireData(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Get unique dates for the filter
  const dates = fireData
    ? [...new Set(fireData.points.map((p) => p.acq_date))].sort()
    : [];

  // Filtered points
  const visiblePoints =
    fireData && selectedDate !== "all"
      ? fireData.points.filter((p) => p.acq_date === selectedDate)
      : fireData?.points ?? [];

  // Initialise Leaflet map
  useEffect(() => {
    if (!mapRef.current || leafletMap.current) return;

    const map = L.map(mapRef.current, {
      center: [30.0, 79.0],
      zoom: 7,
      zoomControl: false,
    });

    // Dark satellite-style tiles
    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://osm.org/copyright">OSM</a>',
        maxZoom: 18,
      }
    ).addTo(map);

    L.control.zoom({ position: "bottomright" }).addTo(map);

    leafletMap.current = map;

    return () => {
      map.remove();
      leafletMap.current = null;
    };
  }, []);

  // Draw fire markers whenever data or filter changes
  useEffect(() => {
    const map = leafletMap.current;
    if (!map || !fireData) return;

    // Clear existing markers
    map.eachLayer((layer) => {
      if (layer instanceof L.CircleMarker) map.removeLayer(layer);
    });

    // Add fire detection markers
    visiblePoints.forEach((pt) => {
      const marker = L.circleMarker([pt.lat, pt.lon], {
        radius: frpToRadius(pt.frp),
        fillColor: frpToColor(pt.frp),
        color: frpToColor(pt.frp),
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0.6,
      });

      marker.bindPopup(
        `<div style="font-family:monospace;font-size:11px;line-height:1.6">
          <strong>Fire Detection</strong><br/>
          Date: ${pt.acq_date}<br/>
          Lat: ${pt.lat.toFixed(4)}, Lon: ${pt.lon.toFixed(4)}<br/>
          Brightness: ${pt.bright_ti4.toFixed(1)} K<br/>
          FRP: ${pt.frp.toFixed(2)} MW<br/>
          Confidence: ${pt.confidence}<br/>
          Day/Night: ${pt.daynight === "D" ? "Day" : "Night"}
        </div>`
      );

      marker.addTo(map);
    });
  }, [fireData, visiblePoints]);

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", borderRadius: "var(--radius)", overflow: "hidden" }}>
      {/* Map container */}
      <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: "absolute", inset: 0, display: "flex", alignItems: "center",
          justifyContent: "center", background: "rgba(13,15,14,0.8)", zIndex: 1000,
        }}>
          <span className="mono" style={{ color: "var(--fg-muted)", fontSize: 12 }}>
            Loading fire detections...
          </span>
        </div>
      )}

      {/* Date filter */}
      {dates.length > 0 && (
        <div style={{
          position: "absolute", top: 12, left: 12, zIndex: 1000,
          background: "rgba(21,25,24,0.9)", backdropFilter: "blur(8px)",
          border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
          padding: "8px 12px", display: "flex", alignItems: "center", gap: 8,
        }}>
          <span className="mono" style={{ fontSize: 9, color: "var(--fg-dim)", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Date
          </span>
          <select
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="mono"
            style={{
              background: "var(--bg)", color: "var(--fg)", border: "1px solid var(--border)",
              borderRadius: 4, padding: "3px 8px", fontSize: 11, cursor: "pointer",
            }}
          >
            <option value="all">All dates ({fireData?.total ?? 0} pts)</option>
            {dates.map((d) => (
              <option key={d} value={d}>
                {d} ({fireData!.points.filter((p) => p.acq_date === d).length} pts)
              </option>
            ))}
          </select>
        </div>
      )}

      {/* FRP Legend */}
      <div style={{
        position: "absolute", bottom: 12, left: 12, zIndex: 1000,
        background: "rgba(21,25,24,0.9)", backdropFilter: "blur(8px)",
        border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
        padding: "8px 12px",
      }}>
        <p className="mono" style={{ fontSize: 8, color: "var(--fg-dim)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 6 }}>
          Fire Radiative Power
        </p>
        <div style={{ display: "flex", gap: 10 }}>
          {[
            { label: "> 15 MW", color: "#ef4444" },
            { label: "5–15 MW", color: "#f97316" },
            { label: "1–5 MW", color: "#eab308" },
            { label: "< 1 MW", color: "#22c55e" },
          ].map((item) => (
            <div key={item.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: item.color, display: "inline-block" }} />
              <span className="mono" style={{ fontSize: 9, color: "var(--fg-muted)" }}>{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Stats badge */}
      <div style={{
        position: "absolute", top: 12, right: 12, zIndex: 1000,
        background: "rgba(21,25,24,0.9)", backdropFilter: "blur(8px)",
        border: "1px solid var(--border)", borderRadius: "var(--radius-sm)",
        padding: "8px 12px",
      }}>
        <span className="mono" style={{ fontSize: 10, color: "var(--fg)" }}>
          {visiblePoints.length} <span style={{ color: "var(--fg-dim)" }}>detections</span>
        </span>
      </div>
    </div>
  );
}
