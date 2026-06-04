import { NextResponse } from "next/server";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

export async function GET() {
  const metricsPath = join(process.cwd(), "..", "reports", "model_metrics.json");

  if (!existsSync(metricsPath)) {
    return NextResponse.json(
      { error: "Model metrics not found. Run train_fire_model.py first." },
      { status: 404 }
    );
  }

  const raw = readFileSync(metricsPath, "utf-8");
  const metrics = JSON.parse(raw);
  return NextResponse.json(metrics);
}
