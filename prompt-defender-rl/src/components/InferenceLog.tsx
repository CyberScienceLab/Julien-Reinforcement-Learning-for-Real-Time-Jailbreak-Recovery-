import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Clock, Shield, Zap } from 'lucide-react';

interface InferenceRequest {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  action: 'ALLOW' | 'BLOCK' | 'REWRITE' | 'ASK' | 'ESCALATE';
  latency: number;
  confidence: number;
  cached: boolean;
  deferred?: boolean;
  moderator_action?: 'ALLOW' | 'BLOCK'; // new
}

interface InferenceLogProps {
  requests: InferenceRequest[];
  getActionColor: (action: string) => string;
}

// Map each action to explicit Tailwind classes (static strings!)
const actionColors: Record<string, string> = {
  ALLOW: "bg-green-100 border-green-300 text-green-700",
  BLOCK: "bg-red-100 border-red-300 text-red-700",
  REWRITE: "bg-yellow-100 border-yellow-300 text-yellow-700",
  ASK: "bg-blue-100 border-blue-300 text-blue-700",
  ESCALATE: "bg-purple-100 border-purple-300 text-purple-700",
};

export const InferenceLog: React.FC<InferenceLogProps> = ({ requests, getActionColor }) => {
  const formatTime = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const truncateText = (text: string, maxLength: number = 80) => {
  if (!text) return '';
  return text.length > maxLength ? text.substring(0, maxLength) + "..." : text;
  };

  // Sort requests by timestamp descending (newest first)
  const sortedRequests = [...requests].sort((a, b) => b.timestamp - a.timestamp);

  return (
    <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <Shield className="h-5 w-5" />
          <span>Real-time Inference Log</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-96 w-full">
          <div className="space-y-3">
            {sortedRequests.slice(0, 20).map((request) => (
              <div
                key={request.id}
                className="p-4 rounded-lg bg-background/50 border border-border/30 space-y-2"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Clock className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs text-muted-foreground">
                      {formatTime(request.timestamp)}
                    </span>
                    <Badge
                      variant="outline"
                      className={`text-xs ${actionColors[request.action] || ""}`}
                    >
                      {request.action}
                    </Badge>
                    {request.moderator_action && (
                      <Badge
                        variant="outline"
                          style={{
                            marginLeft: 4,
                            padding: "2px 6px",
                            borderRadius: 4,
                            background: request.moderator_action === "ALLOW" ? "#d4edda" : "#f8d7da",
                            color: request.moderator_action === "ALLOW" ? "#155724" : "#721c24",
                            fontWeight: "bold",
                            fontSize: "0.85em",
                            border: "1px solid " + (request.moderator_action === "ALLOW" ? "#c3e6cb" : "#f5c6cb")
                          }}
                        >
                          Moderator: {request.moderator_action === "ALLOW" ? "Allow" : "Block"}
                        </Badge>
                    )}
                    {request.cached && (
                      <Badge
                        variant="outline"
                        className="text-xs bg-primary/10 border-primary/30 text-primary flex items-center"
                      >
                        <Zap className="h-3 w-3 mr-1" />
                        CACHED
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                    <span>{request.latency.toFixed(0)}ms</span>
                    <span>•</span>
                    <span>{(request.confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>

                <div className="space-y-1">
                  <div className="text-sm break-words">
                    <span className="text-muted-foreground">Prompt:</span>{" "}
                    <span className="text-foreground">{truncateText(request.prompt ?? '')}</span>
                  </div>
                  <div className="text-sm break-words">
                    <span className="text-muted-foreground">Response:</span>{" "}
                    <span className="text-foreground">{truncateText(request.response ?? '')}</span>
                  </div>
                </div>
              </div>
            ))}

            {requests.length === 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No inference requests yet</p>
                <p className="text-xs">Start the system to see real-time logs</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
};
