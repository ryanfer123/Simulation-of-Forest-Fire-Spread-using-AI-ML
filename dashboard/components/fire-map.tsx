"use client";

import { useRef, useState } from "react";

export type MapLayer = "risk" | "historical" | "spread";

interface FireZone {
  id: string;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  risk: "critical" | "high" | "medium" | "low";
  label: string;
  hectares: number;
}

interface HistoricalFire {
  id: string;
  points: string;
  year: number;
  hectares: number;
  opacity: number;
}

const FIRE_ZONES: FireZone[] = [
  { id: "z1", cx: 210, cy: 160, rx: 55, ry: 40, risk: "critical", label: "Sierra Alta", hectares: 4820 },
  { id: "z2", cx: 420, cy: 220, rx: 70, ry: 50, risk: "high", label: "Pico Rojo", hectares: 3140 },
  { id: "z3", cx: 310, cy: 320, rx: 45, ry: 35, risk: "high", label: "Valle Seco", hectares: 2100 },
  { id: "z4", cx: 560, cy: 160, rx: 40, ry: 30, risk: "medium", label: "Cerro Norte", hectares: 980 },
  { id: "z5", cx: 150, cy: 330, rx: 35, ry: 28, risk: "medium", label: "Bosque Sur", hectares: 750 },
  { id: "z6", cx: 490, cy: 340, rx: 30, ry: 22, risk: "low", label: "Llanura Este", hectares: 320 },
  { id: "z7", cx: 620, cy: 300, rx: 35, ry: 25, risk: "low", label: "Monte Verde", hectares: 410 },
  { id: "z8", cx: 350, cy: 420, rx: 50, ry: 35, risk: "medium", label: "Cañada Seca", hectares: 1250 },
];

const HISTORICAL_FIRES: HistoricalFire[] = [
  { id: "h1", points: "80,120 160,100 200,150 180,220 100,230 60,180", year: 2019, hectares: 6200, opacity: 0.35 },
  { id: "h2", points: "350,80 430,100 470,160 440,220 370,200 330,140", year: 2021, hectares: 8900, opacity: 0.30 },
  { id: "h3", points: "480,250 550,230 600,280 580,360 500,370 460,310", year: 2022, hectares: 5100, opacity: 0.28 },
  { id: "h4", points: "200,350 280,330 300,390 270,440 200,450 170,400", year: 2023, hectares: 3400, opacity: 0.32 },
];

const RISK_COLORS: Record<string, { fill: string; stroke: string; label: string }> = {
  critical: { fill: "rgba(220,38,38,0.25)", stroke: "rgba(220,38,38,0.9)", label: "Critical" },
  high:     { fill: "rgba(234,88,12,0.22)", stroke: "rgba(234,88,12,0.85)", label: "High" },
  medium:   { fill: "rgba(234,179,8,0.20)", stroke: "rgba(234,179,8,0.80)", label: "Medium" },
  low:      { fill: "rgba(34,197,94,0.18)", stroke: "rgba(34,197,94,0.75)", label: "Low" },
};

interface Props {
  activeLayers: Set<MapLayer>;
  simulationActive: boolean;
  simulationStep: number;
}

export default function FireMap({ activeLayers, simulationActive, simulationStep }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredZone, setHoveredZone] = useState<FireZone | null>(null);
  const [tooltip, setTooltip] = useState({ x: 0, y: 0 });

  // Spread simulation rings emanating from the critical zone
  const spreadRings = simulationActive
    ? Array.from({ length: Math.min(simulationStep, 8) }, (_, i) => ({
        cx: 210,
        cy: 160,
        r: 60 + i * 22,
        opacity: Math.max(0.05, 0.45 - i * 0.05),
      }))
    : [];

  // Wind direction indicator
  const windAngle = 38; // degrees

  return (
    <div className="relative w-full h-full overflow-hidden rounded-lg bg-[oklch(0.11_0.01_145)]">
      {/* Terrain base */}
      <svg
        ref={svgRef}
        viewBox="0 0 740 500"
        className="w-full h-full"
        aria-label="Forest fire risk map"
        role="img"
      >
        <defs>
          {/* Forest texture pattern */}
          <pattern id="forest" x="0" y="0" width="20" height="20" patternUnits="userSpaceOnUse">
            <rect width="20" height="20" fill="oklch(0.18 0.04 145)" />
            <circle cx="5" cy="5" r="3" fill="oklch(0.22 0.05 145)" opacity="0.6" />
            <circle cx="15" cy="12" r="2.5" fill="oklch(0.20 0.04 145)" opacity="0.5" />
            <circle cx="10" cy="17" r="2" fill="oklch(0.24 0.06 145)" opacity="0.4" />
          </pattern>
          {/* Dry brush pattern */}
          <pattern id="dry" x="0" y="0" width="16" height="16" patternUnits="userSpaceOnUse">
            <rect width="16" height="16" fill="oklch(0.22 0.04 60)" />
            <line x1="0" y1="8" x2="16" y2="8" stroke="oklch(0.28 0.05 60)" strokeWidth="0.5" opacity="0.4" />
            <line x1="8" y1="0" x2="8" y2="16" stroke="oklch(0.28 0.05 60)" strokeWidth="0.5" opacity="0.4" />
          </pattern>
          <radialGradient id="glowRed" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(220,38,38,0.6)" />
            <stop offset="100%" stopColor="rgba(220,38,38,0)" />
          </radialGradient>
          <radialGradient id="glowOrange" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="rgba(234,88,12,0.5)" />
            <stop offset="100%" stopColor="rgba(234,88,12,0)" />
          </radialGradient>
          <filter id="blur4">
            <feGaussianBlur stdDeviation="4" />
          </filter>
          <filter id="blur8">
            <feGaussianBlur stdDeviation="8" />
          </filter>
        </defs>

        {/* Base terrain */}
        <rect width="740" height="500" fill="url(#forest)" />

        {/* Elevation contour lines */}
        {[50, 120, 200, 280, 360, 430].map((y, i) => (
          <path
            key={i}
            d={`M0,${y} Q185,${y - 15 + i * 3} 370,${y + 10 - i * 2} Q555,${y - 8 + i * 4} 740,${y + 5}`}
            fill="none"
            stroke="oklch(0.30 0.03 145)"
            strokeWidth="0.8"
            opacity="0.4"
          />
        ))}

        {/* Rivers */}
        <path
          d="M0,250 Q80,240 140,260 Q200,280 250,265 Q300,250 370,270 Q450,290 520,275 Q600,260 680,280 L740,275"
          fill="none"
          stroke="oklch(0.45 0.12 220)"
          strokeWidth="2.5"
          opacity="0.55"
        />
        <path
          d="M300,0 Q310,80 290,160 Q270,240 285,320 Q300,400 310,500"
          fill="none"
          stroke="oklch(0.45 0.12 220)"
          strokeWidth="1.8"
          opacity="0.4"
        />

        {/* Roads */}
        <path
          d="M0,380 L740,350"
          fill="none"
          stroke="oklch(0.35 0.01 60)"
          strokeWidth="2"
          strokeDasharray="8,4"
          opacity="0.5"
        />
        <path
          d="M370,0 L360,500"
          fill="none"
          stroke="oklch(0.35 0.01 60)"
          strokeWidth="1.5"
          strokeDasharray="6,4"
          opacity="0.4"
        />

        {/* Historical fire overlays */}
        {activeLayers.has("historical") &&
          HISTORICAL_FIRES.map((hf) => (
            <g key={hf.id}>
              <polygon
                points={hf.points}
                fill={`rgba(120,50,10,${hf.opacity})`}
                stroke="rgba(160,60,10,0.6)"
                strokeWidth="1.5"
                strokeDasharray="4,3"
              />
              <text
                x={hf.points.split(" ")[0].split(",")[0]}
                y={Number(hf.points.split(" ")[0].split(",")[1]) - 6}
                fontSize="9"
                fill="rgba(200,120,60,0.9)"
                fontFamily="monospace"
              >
                {hf.year} · {(hf.hectares / 1000).toFixed(1)}k ha
              </text>
            </g>
          ))}

        {/* Spread simulation rings */}
        {activeLayers.has("spread") &&
          spreadRings.map((ring, i) => (
            <ellipse
              key={i}
              cx={ring.cx + i * 4}
              cy={ring.cy + i * 6}
              rx={ring.r}
              ry={ring.r * 0.75}
              fill={`rgba(220,38,38,${ring.opacity * 0.3})`}
              stroke={`rgba(220,38,38,${ring.opacity})`}
              strokeWidth="1.5"
              filter="url(#blur4)"
            />
          ))}

        {/* Risk zone glows (blur halo) */}
        {activeLayers.has("risk") &&
          FIRE_ZONES.filter((z) => z.risk === "critical" || z.risk === "high").map((zone) => (
            <ellipse
              key={`glow-${zone.id}`}
              cx={zone.cx}
              cy={zone.cy}
              rx={zone.rx + 30}
              ry={zone.ry + 22}
              fill={zone.risk === "critical" ? "url(#glowRed)" : "url(#glowOrange)"}
              filter="url(#blur8)"
            />
          ))}

        {/* Risk zones */}
        {activeLayers.has("risk") &&
          FIRE_ZONES.map((zone) => {
            const color = RISK_COLORS[zone.risk];
            const isHovered = hoveredZone?.id === zone.id;
            return (
              <g
                key={zone.id}
                className="cursor-pointer"
                onMouseEnter={(e) => {
                  setHoveredZone(zone);
                  const svg = svgRef.current;
                  if (svg) {
                    const rect = svg.getBoundingClientRect();
                    const scaleX = rect.width / 740;
                    const scaleY = rect.height / 500;
                    setTooltip({ x: zone.cx * scaleX, y: zone.cy * scaleY });
                  }
                }}
                onMouseLeave={() => setHoveredZone(null)}
              >
                <ellipse
                  cx={zone.cx}
                  cy={zone.cy}
                  rx={zone.rx + (isHovered ? 4 : 0)}
                  ry={zone.ry + (isHovered ? 3 : 0)}
                  fill={color.fill}
                  stroke={color.stroke}
                  strokeWidth={isHovered ? 2 : 1.5}
                  className="transition-all duration-150"
                />
                {/* Pulsing center dot for critical/high */}
                {(zone.risk === "critical" || zone.risk === "high") && (
                  <circle
                    cx={zone.cx}
                    cy={zone.cy}
                    r={4}
                    fill={color.stroke}
                    opacity="0.9"
                  />
                )}
                <text
                  x={zone.cx}
                  y={zone.cy - zone.ry - 5}
                  textAnchor="middle"
                  fontSize="9.5"
                  fill={color.stroke}
                  fontFamily="monospace"
                  fontWeight="600"
                >
                  {zone.label}
                </text>
              </g>
            );
          })}

        {/* Wind direction arrow */}
        <g transform={`translate(680,50) rotate(${windAngle})`}>
          <circle cx="0" cy="0" r="22" fill="oklch(0.20 0.01 30)" stroke="oklch(0.30 0.01 30)" strokeWidth="1" />
          <polygon points="0,-14 5,6 0,2 -5,6" fill="oklch(0.68 0.21 40)" />
          <text x="0" y="32" textAnchor="middle" fontSize="8" fill="oklch(0.60 0.02 50)" fontFamily="monospace">
            WIND
          </text>
          <text x="0" y="41" textAnchor="middle" fontSize="8" fill="oklch(0.60 0.02 50)" fontFamily="monospace">
            18 km/h
          </text>
        </g>

        {/* Scale bar */}
        <g transform="translate(20,470)">
          <line x1="0" y1="0" x2="80" y2="0" stroke="oklch(0.55 0.01 60)" strokeWidth="1.5" />
          <line x1="0" y1="-4" x2="0" y2="4" stroke="oklch(0.55 0.01 60)" strokeWidth="1.5" />
          <line x1="80" y1="-4" x2="80" y2="4" stroke="oklch(0.55 0.01 60)" strokeWidth="1.5" />
          <text x="40" y="-6" textAnchor="middle" fontSize="8" fill="oklch(0.55 0.01 60)" fontFamily="monospace">
            10 km
          </text>
        </g>

        {/* Coordinates */}
        <text x="720" y="494" textAnchor="end" fontSize="7.5" fill="oklch(0.40 0.01 60)" fontFamily="monospace">
          40.42°N  3.69°W
        </text>
      </svg>

      {/* Tooltip */}
      {hoveredZone && (
        <div
          className="absolute z-20 pointer-events-none bg-card border border-border rounded px-3 py-2 text-xs font-mono shadow-lg"
          style={{ left: tooltip.x + 12, top: tooltip.y - 48 }}
        >
          <div className="font-bold text-foreground">{hoveredZone.label}</div>
          <div className="flex gap-3 mt-1">
            <span style={{ color: RISK_COLORS[hoveredZone.risk].stroke }}>
              {RISK_COLORS[hoveredZone.risk].label} Risk
            </span>
            <span className="text-muted-foreground">{hoveredZone.hectares.toLocaleString()} ha</span>
          </div>
        </div>
      )}

      {/* Map overlay gradient for depth */}
      <div className="absolute inset-0 pointer-events-none rounded-lg ring-1 ring-inset ring-border/40" />
    </div>
  );
}
