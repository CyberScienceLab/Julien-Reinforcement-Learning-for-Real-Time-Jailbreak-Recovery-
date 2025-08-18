import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, Zap } from 'lucide-react';

interface InferenceRequest {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  action: 'ALLOW' | 'BLOCK' | 'REWRITE' | 'ASK' | 'ESCALATE';
  latency: number;
  confidence: number;
  cached: boolean;
}

interface LatencyChartProps {
  requests: InferenceRequest[];
}

export const LatencyChart: React.FC<LatencyChartProps> = ({ requests }) => {
  // Prepare data for the chart - last 30 requests
  // Reverse by prompt number (latest first)
  const chartData = [...requests]
    .slice(-30)
    .map((req, idx, arr) => ({
      index: arr.length - idx,
      latency: Math.round(req.latency),
      cached: req.cached,
      timestamp: new Date(req.timestamp).toLocaleTimeString(),
      action: req.action
    }))
    .reverse();

  const avgLatency = chartData.length > 0 
    ? Math.round(chartData.reduce((sum, item) => sum + item.latency, 0) / chartData.length)
    : 0;

  const p99Target = 200; // P99 target of 200ms

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium">Request #{label}</p>
          <p className="text-xs text-muted-foreground">{data.timestamp}</p>
          <p className="text-sm">
            Latency: <span className="font-bold text-primary">{data.latency}ms</span>
          </p>
          <p className="text-sm">
            Action: <span className="font-bold">{data.action}</span>
          </p>
          {data.cached && (
            <div className="flex items-center space-x-1 text-xs text-primary">
              <Zap className="h-3 w-3" />
              <span>Cached</span>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5" />
            <span>Latency Monitoring</span>
          </div>
          <div className="text-sm text-muted-foreground">
            Avg: {avgLatency}ms
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {chartData.length > 0 ? (
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis 
                  dataKey="index"
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={{ stroke: 'hsl(var(--border))' }}
                />
                <YAxis 
                  tick={{ fontSize: 12, fill: 'hsl(var(--muted-foreground))' }}
                  axisLine={{ stroke: 'hsl(var(--border))' }}
                  label={{ 
                    value: 'Latency (ms)', 
                    angle: -90, 
                    position: 'insideLeft',
                    style: { textAnchor: 'middle', fill: 'hsl(var(--muted-foreground))' }
                  }}
                />
                <Tooltip content={<CustomTooltip />} />
                
                {/* P99 target line */}
                <ReferenceLine 
                  y={p99Target} 
                  stroke="hsl(var(--action-block))" 
                  strokeDasharray="5 5"
                  label={{ value: "P99 Target (200ms)", position: "top" }}
                />
                
                {/* Average line */}
                <ReferenceLine 
                  y={avgLatency} 
                  stroke="hsl(var(--primary))" 
                  strokeDasharray="3 3"
                  label={{ value: `Avg (${avgLatency}ms)`, position: "top" }}
                />
                
                <Line
                  type="monotone"
                  dataKey="latency"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  dot={(props) => {
                    const { cx, cy, payload, index } = props;
                    return (
                      <circle
                        key={payload?.id || index}
                        cx={cx}
                        cy={cy}
                        r={payload?.cached ? 6 : 4}
                        fill={payload?.cached ? "hsl(var(--action-allow))" : "hsl(var(--primary))"}
                        stroke={payload?.cached ? "hsl(var(--action-allow))" : "hsl(var(--primary))"}
                        strokeWidth={2}
                      />
                    );
                  }}
                  activeDot={{ r: 6, stroke: "hsl(var(--primary))", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No latency data available</p>
              <p className="text-xs">Process requests to see latency trends</p>
            </div>
          </div>
        )}
        
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-primary"></div>
              <span>Standard</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-action-allow"></div>
              <span>Cached</span>
            </div>
          </div>
          <span>Last 30 requests</span>
        </div>
      </CardContent>
    </Card>
  );
};