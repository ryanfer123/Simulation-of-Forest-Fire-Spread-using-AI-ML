import { NextResponse } from "next/server";
import { existsSync, readFileSync } from "fs";
import { join } from "path";

interface FirePoint {
  lat: number;
  lon: number;
  bright_ti4: number;
  frp: number;
  confidence: string;
  acq_date: string;
  daynight: string;
}

export async function GET() {
  const csvPath = join(process.cwd(), "..", "data", "suomi_viirs_7d.csv");

  if (!existsSync(csvPath)) {
    return NextResponse.json(
      { error: "FIRMS data not found. Run fetch_firms_data.py first." },
      { status: 404 }
    );
  }

  // Read and filter to Uttarakhand bounding box
  const bounds = { minLon: 77.0, minLat: 28.5, maxLon: 81.0, maxLat: 31.5 };
  const raw = readFileSync(csvPath, "utf-8");
  const lines = raw.split("\n");
  const header = lines[0].split(",");

  const latIdx = header.indexOf("latitude");
  const lonIdx = header.indexOf("longitude");
  const brightIdx = header.indexOf("bright_ti4");
  const frpIdx = header.indexOf("frp");
  const confIdx = header.indexOf("confidence");
  const dateIdx = header.indexOf("acq_date");
  const dnIdx = header.indexOf("daynight");

  const points: FirePoint[] = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(",");
    if (cols.length < header.length) continue;

    const lat = parseFloat(cols[latIdx]);
    const lon = parseFloat(cols[lonIdx]);

    if (lat < bounds.minLat || lat > bounds.maxLat || lon < bounds.minLon || lon > bounds.maxLon) {
      continue;
    }

    points.push({
      lat,
      lon,
      bright_ti4: parseFloat(cols[brightIdx]) || 0,
      frp: parseFloat(cols[frpIdx]) || 0,
      confidence: cols[confIdx] || "nominal",
      acq_date: cols[dateIdx] || "",
      daynight: cols[dnIdx] || "D",
    });
  }

  return NextResponse.json({
    total: points.length,
    bounds,
    points,
  });
}
