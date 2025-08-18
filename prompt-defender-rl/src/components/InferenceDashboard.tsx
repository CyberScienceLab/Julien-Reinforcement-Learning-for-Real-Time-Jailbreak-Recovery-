import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { MetricsGrid } from './MetricsGrid';
import { InferenceLog } from './InferenceLog';
import { ActionDistribution } from './ActionDistribution';
import { LatencyChart } from './LatencyChart';
import { Shield, Activity, Zap, AlertTriangle, Play, Pause, Send } from 'lucide-react';
import { useInferenceAPI, InferenceRequest } from '@/hooks/useInferenceAPI';

const InferenceDashboard: React.FC = () => {
  const [isRunning, setIsRunning] = useState(true);
  const [testPrompt, setTestPrompt] = useState('');
  const [showModDecision, setShowModDecision] = useState(false);
  const [pendingResult, setPendingResult] = useState<any>(null);
  const [modStartTime, setModStartTime] = useState<number>(0);
  const [showAskModal, setShowAskModal] = useState(false);
  const [askPrompt, setAskPrompt] = useState('');
  const [cachedRequests, setCachedRequests] = useState<any[]>([]);
  const [requests, setRequests] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<any>({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch processed and cached prompts from backend
  const fetchInferenceHistory = async () => {
    setIsLoading(true);
    try {
      const resp = await fetch('http://localhost:8000/inference/history');
      const data = await resp.json();
      setRequests(prev => {
        // Merge moderator actions from previous state into new history
        const prevById = Object.fromEntries(prev.map(r => [r.id, r]));
        const newProcessed = (data.processed || []).map(r => {
          if (prevById[r.id] && prevById[r.id].moderator_action) {
            return {
              ...r,
              moderator_action: prevById[r.id].moderator_action,
              response: prevById[r.id].response,
              latency: prevById[r.id].latency
            };
          }
          return r;
        });
        // Also, if a moderator decision was made but the backend hasn't reflected it yet, keep the entry
  const missing = Object.values(prevById).filter(r => (r as any).moderator_action && !newProcessed.find(nr => (nr as any).id === (r as any).id));
  return [...newProcessed, ...missing];
      });
      setCachedRequests(data.cached || []);
    } catch (err) {
      setError('Failed to fetch inference history');
    }
    setIsLoading(false);
  };

  // Fetch metrics
  const fetchMetrics = async () => {
    try {
      const resp = await fetch('http://localhost:8000/metrics');
      const data = await resp.json();
      setMetrics(data);
    } catch (err) {
      setError('Failed to fetch metrics');
    }
  };

  useEffect(() => {
    fetchInferenceHistory();
    fetchMetrics();
    const interval = setInterval(() => {
      fetchInferenceHistory();
      fetchMetrics();
    }, 10000);
    return () => clearInterval(interval);
  }, []);

  // Removed auto-inference useEffect. Inference now only occurs on button press.
  const handleModeratorDecision = async (decision: 'ALLOW' | 'BLOCK') => {
    if (!pendingResult || !modStartTime) return;
    const latency = pendingResult.latency + (Date.now() - modStartTime);
    try {
      await fetch('http://localhost:8000/moderator_decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: pendingResult.id,
          prompt: pendingResult.prompt,
          original_action: pendingResult.action,
          moderator_action: decision,
          latency,
          timestamp: Date.now(),
        }),
      });
    } catch {}
    // Update only the relevant entry, preserving the rest
    setRequests(prev => prev.map(r =>
      r.id === pendingResult.id
        ? { ...r, response: `Moderator decision: ${decision}`, moderator_action: decision, latency }
        : r
    ));
    setPendingResult(null);
    setModStartTime(0);
    setShowModDecision(false);
    // Do NOT prepend or replace the array, just update the entry
    fetchInferenceHistory();
    fetchMetrics();
  };

  const handleTestSubmit = async () => {
    if (testPrompt.trim() === '') return;
    const actualPrompt = testPrompt.trim();
    try {
      const resp = await fetch('http://localhost:8000/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: actualPrompt })
      });
      const result = await resp.json();
      setTestPrompt('');
      if (result.cached) {
        setCachedRequests(prev => [result, ...prev]);
      } else {
        if (result.action === 'ESCALATE') {
          const escalationTime = Date.now();
          setRequests(prev => [
            {
              ...result,
              response: 'Request escalated for human review.',
              latency: result.latency,
              action: 'ESCALATE',
            },
            ...prev
          ]);
          setModStartTime(escalationTime);
          setPendingResult(result);
          setShowModDecision(true);
        } else if (result.action === 'ASK') {
          setRequests(prev => [
            {
              ...result,
              response: 'Model requested clarification. Please enter a new prompt.',
              latency: result.latency,
              action: 'ASK',
            },
            ...prev
          ]);
          setPendingResult(result);
          setShowAskModal(true);
        } else {
          setRequests(prev => [result, ...prev]);
        }
      }
      fetchInferenceHistory();
      fetchMetrics();
    } catch (err) {
      setError('Failed to submit test inference');
    }
  };


  const getActionColor = (action: string) => {
    switch (action) {
      case 'ALLOW': return 'action-allow';
      case 'BLOCK': return 'action-block';
      case 'REWRITE': return 'action-rewrite';
      case 'ASK': return 'action-ask';
      case 'ESCALATE': return 'action-escalate';
      default: return 'muted';
    }
  };

  const getThreatLevelColor = (level: string) => {
    switch (level) {
      case 'HIGH': return 'action-block';
      case 'MEDIUM': return 'action-rewrite';
      case 'LOW': return 'action-allow';
      default: return 'muted';
    }
  };


  // Cache management handlers
  const handleProcessCache = async () => {
    try {
      const resp = await fetch('http://localhost:8000/process_cache', { method: 'POST' });
      const data = await resp.json();
      // If any processed prompt is ASK or ESCALATE, open the modal for the first one
      if (Array.isArray(data.processed) && data.processed.length > 0) {
        let handled = false;
        for (const result of data.processed) {
          if (result.action === 'ESCALATE' && !handled) {
            setModStartTime(Date.now());
            setPendingResult(result);
            setShowModDecision(true);
            handled = true;
          } else if (result.action === 'ASK' && !handled) {
            setModStartTime(Date.now());
            setPendingResult(result);
            setShowAskModal(true);
            handled = true;
          }
        }
        // Add all processed prompts to requests
        setRequests(prev => [...data.processed, ...prev]);
      }
      alert(`Processed ${data.count} cached prompts.`);
      fetchInferenceHistory();
      fetchMetrics();
    } catch (err) {
      alert('Failed to process cache');
    }
  };

  const handleClearCache = async () => {
    try {
      const resp = await fetch('http://localhost:8000/clear_cache', { method: 'POST' });
      await resp.json();
      setCachedRequests([]);
      alert('Cache cleared.');
      fetchInferenceHistory();
      fetchMetrics();
    } catch (err) {
      alert('Failed to clear cache');
    }
  };

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Shield className="h-8 w-8 text-primary" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              RL Defense Layer
            </h1>
          </div>
          <Badge variant="outline" className="text-xs">
            v1.0.0-beta
          </Badge>
        </div>
        
        <div className="flex items-center space-x-4">
          <Button
            variant={isRunning ? "secondary" : "default"}
            onClick={() => setIsRunning(!isRunning)}
            className="flex items-center space-x-2"
          >
            {isRunning ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            <span>{isRunning ? 'Pause' : 'Start'}</span>
          </Button>
          
          <div className="flex items-center space-x-2">
            <div className={`h-3 w-3 rounded-full ${isRunning ? 'bg-action-allow animate-pulse' : 'bg-muted'}`} />
            <span className="text-sm text-muted-foreground">
              {isRunning ? 'Live' : 'Stopped'}
            </span>
          </div>
        </div>
      </div>

            {/* Show prompt for ESCALATE cases */}
            {pendingResult && pendingResult.action === 'ESCALATE' && (
              <div className="mb-4">
                <div className="text-sm text-muted-foreground mb-1">Prompt:</div>
                <div className="text-base text-foreground break-words p-2 rounded bg-background/80 border border-border/30">
                  {pendingResult.prompt}
                </div>
              </div>
            )}
      {/* Error Alert */}
      {error && (
        <Alert className="border-action-block bg-action-block/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {error}
          </AlertDescription>
        </Alert>
      )}

      {/* Threat Level Alert */}
      {metrics.threatLevel === 'HIGH' && (
        <Alert className="border-action-block bg-action-block/10">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            High threat activity detected. Enhanced monitoring is active.
          </AlertDescription>
        </Alert>
      )}

      {/* Test Interface */}
      <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Send className="h-5 w-5" />
            <span>Test Inference</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="test-prompt">Test Prompt (Optional)</Label>
              <Input
                id="test-prompt"
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                placeholder="Enter a test prompt..."
              />
            </div>
            <Button 
              onClick={handleTestSubmit}
              disabled={isLoading}
              className="flex items-center space-x-2"
            >
              <Send className="h-4 w-4" />
              <span>{isLoading ? 'Submitting...' : 'Submit Test Inference'}</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Cache Management Buttons */}
      <div className="flex space-x-4 mb-4">
        <Button variant="outline" onClick={handleProcessCache}>
          Process Cache ({cachedRequests.length})
        </Button>
        <Button variant="outline" onClick={handleClearCache}>
          Clear Cache
        </Button>
      </div>

      {/* Main Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Requests</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(metrics.totalRequests ?? 0).toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">+{requests.length} in session</p>
          </CardContent>
        </Card>

        <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.avgLatency}ms</div>
            <Progress 
              value={Math.max(0, 100 - (metrics.avgLatency / 2))} 
              className="h-2 mt-2"
            />
          </CardContent>
        </Card>

        <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{metrics.cacheHitRate}%</div>
            <Progress value={metrics.cacheHitRate} className="h-2 mt-2" />
          </CardContent>
        </Card>

        <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Threat Level</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold text-${getThreatLevelColor(metrics.threatLevel)}`}>
              {metrics.threatLevel}
            </div>
            <p className="text-xs text-muted-foreground">Real-time assessment</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts and Logs Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <LatencyChart requests={requests} />
          <InferenceLog requests={requests} getActionColor={getActionColor} />
          {/* Cached prompts log */}
          {cachedRequests.length > 0 && (
            <div className="mt-6">
              <Card className="border-border/50 bg-gradient-to-br from-card to-card/50">
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Zap className="h-5 w-5 text-action-allow" />
                    <span>Cached Prompts Log</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <InferenceLog requests={cachedRequests} getActionColor={getActionColor} />
                </CardContent>
              </Card>
            </div>
          )}
        </div>
        <div className="space-y-6">
          <ActionDistribution requests={[...requests, ...cachedRequests]} />
          <MetricsGrid requests={requests} />
        </div>
      </div>
      {showModDecision && pendingResult && pendingResult.action === 'ESCALATE' && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6 w-full max-w-md space-y-4">
            <div className="mb-4">
              <div className="text-sm text-muted-foreground mb-1">Prompt:</div>
              <div className="text-base text-foreground break-words p-2 rounded bg-background/80 border border-border/30">
                {pendingResult.prompt}
              </div>
            </div>
            <h2 className="text-lg font-semibold text-red-600">Moderator Review Required</h2>
            <p className="text-sm text-muted-foreground mb-4">
              This request was escalated. Please select an action:
            </p>

            <div className="flex justify-center space-x-4">
              <Button
                className="bg-green-500 hover:bg-green-600"
                onClick={() => handleModeratorDecision('ALLOW')}
              >
                Allow
              </Button>
              <Button
                className="bg-red-500 hover:bg-red-600"
                onClick={() => handleModeratorDecision('BLOCK')}
              >
                Block
              </Button>
            </div>
          </div>
        </div>
      )}
      {showAskModal && (
        <div className="fixed inset-0 z-50 bg-black bg-opacity-50 flex items-center justify-center">
          <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6 w-full max-w-md space-y-4">
            <h2 className="text-lg font-semibold text-blue-600">Clarification Requested</h2>
            <p className="text-sm text-muted-foreground mb-4">
              The model requested clarification. Please enter a new prompt:
            </p>
              <div className="mb-4">
                <span className="text-muted-foreground">Prompt:</span>
                <div className="rounded p-2 mt-1 text-sm break-words" style={{ background: '#222', color: '#fff' }}>{pendingResult?.prompt}</div>
              </div>
              <textarea
                className="w-full border rounded p-2 mb-4"
                rows={3}
                value={askPrompt}
                onChange={e => setAskPrompt(e.target.value)}
                placeholder="Enter clarification..."
                style={{ background: '#222', color: '#fff' }}
              />
            <div className="flex justify-center space-x-4 mt-4">
              <Button
                className="bg-blue-500 hover:bg-blue-600"
                onClick={async () => {
                  if (askPrompt.trim() !== '') {
                    // Calculate latency for ASK
                    const latency = pendingResult.latency + (Date.now() - modStartTime);
                    const resp = await fetch('http://localhost:8000/ask_clarification', {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        id: pendingResult.id,
                        original_prompt: pendingResult.prompt,
                        clarification: askPrompt.trim(),
                        latency,
                        timestamp: Date.now(),
                      })
                    });
                    let result = await resp.json();
                    // If the model returns ASK again, escalate instead
                    if (result.action === 'ASK') {
                      result = {
                        ...result,
                        action: 'ESCALATE',
                        response: 'Request escalated for human review after repeated clarification.'
                      };
                    }
                    // Remove the old ASK entry and add the new result
                    setRequests(prev => [result, ...prev.filter(r => r.id !== pendingResult.id)]);
                    setShowAskModal(false);
                    setAskPrompt('');
                    setPendingResult(null);
                    setModStartTime(0);
                    // If the result is ESCALATE, open moderator popup for that result
                    if (result.action === 'ESCALATE') {
                      setPendingResult(result);
                      setModStartTime(Date.now());
                      setShowModDecision(true);
                    }
                    fetchInferenceHistory();
                    fetchMetrics();
                  }
                }}
              >
                Submit Clarification
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setShowAskModal(false);
                  setAskPrompt('');
                  setPendingResult(null);
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
};

export default InferenceDashboard;