import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ChartPoint } from "../types";
import { formatTimestamp } from "../metrics";

interface ForecastChartProps {
  data: ChartPoint[];
  showComponents: boolean;
  historyTail: number;
}

export function ForecastChart({ data, showComponents, historyTail }: ForecastChartProps) {
  const history = data.filter((d) => d.kind === "history");
  const forecast = data.filter((d) => d.kind === "forecast");
  const tailHistory = history.slice(-historyTail);
  const chartData = [...tailHistory, ...forecast];

  const firstForecastIdx = chartData.findIndex((d) => d.kind === "forecast");
  const dividerLabel =
    firstForecastIdx >= 0 ? chartData[firstForecastIdx]?.label : undefined;

  return (
    <div className="chart-wrap">
      <ResponsiveContainer width="100%" height={420}>
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="label"
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            interval="preserveStartEnd"
            minTickGap={40}
          />
          <YAxis
            domain={[0, "auto"]}
            tick={{ fill: "#94a3b8", fontSize: 11 }}
            label={{ value: "CPU %", angle: -90, position: "insideLeft", fill: "#94a3b8" }}
          />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
            labelStyle={{ color: "#e2e8f0" }}
            formatter={(value: number, name: string) => [
              typeof value === "number" ? value.toFixed(3) : value,
              name,
            ]}
            labelFormatter={(_, payload) => {
              const ts = payload?.[0]?.payload?.timestamp as string | undefined;
              return ts ? formatTimestamp(ts) : "";
            }}
          />
          <Legend />

          {dividerLabel && (
            <ReferenceLine x={dividerLabel} stroke="#64748b" strokeDasharray="4 4" />
          )}

          <Line
            type="monotone"
            dataKey="actual"
            name="Actual / History"
            stroke="#38bdf8"
            dot={false}
            strokeWidth={2}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="predicted"
            name="Predicted"
            stroke="#f97316"
            dot={false}
            strokeWidth={2}
            connectNulls={false}
          />
          {showComponents && (
            <>
              <Line
                type="monotone"
                dataKey="prophet"
                name="Prophet"
                stroke="#a78bfa"
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="5 3"
              />
              <Line
                type="monotone"
                dataKey="residual"
                name="GRU residual"
                stroke="#4ade80"
                dot={false}
                strokeWidth={1.5}
                strokeDasharray="5 3"
              />
            </>
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
