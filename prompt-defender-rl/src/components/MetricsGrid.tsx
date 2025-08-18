import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Cpu, Database, Network, Timer } from 'lucide-react';

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

interface MetricsGridProps {
  requests: InferenceRequest[];
}

export const MetricsGrid: React.FC<MetricsGridProps> = ({ requests }) => {
  const recentRequests = requests.slice(0, 20);
  
  const totalRequests = recentRequests.length;
  const avgLatency = recentRequests.length > 0 
    ? recentRequests.reduce((sum, req) => sum + req.latency, 0) / recentRequests.length 
    : 0;
  const cacheHitRate = recentRequests.length > 0 
    ? (recentRequests.filter(req => req.cached).length / recentRequests.length) * 100 
    : 0;
  const highConfidenceRequests = recentRequests.filter(req => req.confidence > 0.8).length;
  const avgConfidence = recentRequests.length > 0 
    ? Math.round(recentRequests.reduce((sum, req) => sum + req.confidence, 0) / recentRequests.length * 1000) / 1000
    : 0;
  const p99Latency = recentRequests.length > 0
    ? Math.round(recentRequests.map(r => r.latency).sort((a, b) => b - a)[Math.floor(recentRequests.length * 0.01)] || 0)
    : 0;
    
  const modelLoad = Math.round(Math.random() * 30 + 40); // Mock CPU usage
  const memoryUsage = Math.round(Math.random() * 20 + 60); // Mock memory usage
  const gpuUtilization = Math.round(Math.random() * 40 + 30); // Mock GPU usage

  return (
    <div className="space-y-4">
      <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Model Confidence</CardTitle>
          <Cpu className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{avgConfidence}</div>
          <Progress value={avgConfidence} className="h-2 mt-2" />
          <p className="text-xs text-muted-foreground mt-1">Avg over last 20 requests</p>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">High Confidence</CardTitle>
          <Database className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-action-allow">{highConfidenceRequests}</div>
          <Progress value={recentRequests.length > 0 ? (highConfidenceRequests / recentRequests.length) * 100 : 0} className="h-2 mt-2" />
          <p className="text-xs text-muted-foreground mt-1">Requests with {'>'}0.8 confidence</p>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">P99 Latency</CardTitle>
          <Timer className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{p99Latency}ms</div>
          <Progress 
            value={Math.max(0, 100 - (p99Latency / 2))} 
            className="h-2 mt-2" 
          />
          <p className="text-xs text-muted-foreground mt-1">99th percentile</p>
        </CardContent>
      </Card>

      <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Model Load</CardTitle>
          <Network className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{modelLoad}%</div>
          <Progress value={modelLoad} className="h-2 mt-2" />
          <p className="text-xs text-muted-foreground mt-1">CPU utilization</p>
        </CardContent>
      </Card>
    </div>
  );
};