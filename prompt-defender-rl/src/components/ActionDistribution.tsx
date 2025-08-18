import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Shield, Check, X, Edit, MessageCircle, AlertTriangle } from 'lucide-react';

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

interface ActionDistributionProps {
  requests: InferenceRequest[];
}

export const ActionDistribution: React.FC<ActionDistributionProps> = ({ requests }) => {
  const actionCounts = requests.reduce((acc, req) => {
    acc[req.action] = (acc[req.action] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const actionColors = {
    ALLOW: 'hsl(var(--action-allow))',
    BLOCK: 'hsl(var(--action-block))',
    REWRITE: 'hsl(var(--action-rewrite))',
    ASK: 'hsl(var(--action-ask))',
    ESCALATE: 'hsl(var(--action-escalate))'
  };

  const actionIcons = {
    ALLOW: Check,
    BLOCK: X,
    REWRITE: Edit,
    ASK: MessageCircle,
    ESCALATE: AlertTriangle
  };

  const chartData = Object.entries(actionCounts).map(([action, count]) => ({
    name: action,
    value: count,
    color: actionColors[action as keyof typeof actionColors]
  }));

  const total = Object.values(actionCounts).reduce((sum, count) => sum + count, 0);

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Shield className="h-5 w-5" />
          <span>Action Distribution</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {total > 0 ? (
          <div className="space-y-4">
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={chartData}
                    cx="50%"
                    cy="50%"
                    innerRadius={40}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {chartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: number) => [value, 'Count']}
                    labelStyle={{ color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px',
                      color: '#fff' // Ensures readable text
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            
            <div className="space-y-3">
              {Object.entries(actionCounts).map(([action, count]) => {
                const Icon = actionIcons[action as keyof typeof actionIcons];
                const percentage = Math.round((count / total) * 100);
                
                return (
                  <div key={action} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-2">
                        <Icon className="h-4 w-4" style={{ color: actionColors[action as keyof typeof actionColors] }} />
                        <span>{action}</span>
                      </div>
                      <span className="text-muted-foreground">{count} ({percentage}%)</span>
                    </div>
                    <Progress 
                      value={percentage} 
                      className="h-2"
                      style={{ 
                        '--progress-background': actionColors[action as keyof typeof actionColors] 
                      } as React.CSSProperties}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No actions recorded yet</p>
            <p className="text-xs">Start processing requests to see distribution</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};