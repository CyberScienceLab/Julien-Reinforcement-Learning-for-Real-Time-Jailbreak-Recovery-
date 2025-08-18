import { useState, useEffect, useCallback } from 'react';

export interface InferenceRequest {
  id: string;
  timestamp: number;
  prompt: string;
  response: string;
  action: 'ALLOW' | 'BLOCK' | 'REWRITE' | 'ASK' | 'ESCALATE';
  latency: number;
  confidence: number;
  cached: boolean;
}

export interface Metrics {
  totalRequests: number;
  avgLatency: number;
  cacheHitRate: number;
  threatLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  actionCounts: {
    ALLOW: number;
    BLOCK: number;
    REWRITE: number;
    ASK: number;
    ESCALATE: number;
  };
}

const API_BASE_URL = 'http://localhost:8000';

export const useInferenceAPI = () => {
  const [requests, setRequests] = useState<InferenceRequest[]>([]);
  const [metrics, setMetrics] = useState<Metrics>({
    totalRequests: 0,
    avgLatency: 0,
    cacheHitRate: 0,
    threatLevel: 'LOW',
    actionCounts: {
      ALLOW: 0,
      BLOCK: 0,
      REWRITE: 0,
      ASK: 0,
      ESCALATE: 0
    }
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch inference history
  const fetchInferenceHistory = useCallback(async () => {
    try {
      console.log('🔄 Fetching inference history...');
      const response = await fetch(`${API_BASE_URL}/inference/history`);
      if (!response.ok) throw new Error('Failed to fetch inference history');
      
      const data = await response.json();
      console.log('📜 Received inference history:', data);
      setRequests(data.requests || []);
    } catch (err) {
      console.error('Failed to fetch inference history:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  // Fetch metrics
  const fetchMetrics = useCallback(async () => {
    try {
      console.log('🔄 Fetching metrics...');
      const response = await fetch(`${API_BASE_URL}/metrics`);
      if (!response.ok) throw new Error('Failed to fetch metrics');
      
      const data = await response.json();
      console.log('📊 Received metrics:', data);
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  // Submit inference request
  const submitInference = useCallback(async (prompt: string) => {
    setIsLoading(true);
    setError(null);
    
    // Validate input before sending
    if (!prompt || prompt.trim() === '') {
      setError('Empty prompt provided');
      setIsLoading(false);
      return null;
    }
    
    try {
      const response = await fetch(`${API_BASE_URL}/inference`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Failed to submit inference request: ${response.status} - ${errorText}`);
      }
      
      const inferenceResult = await response.json();
      
      // Check if backend returned an error
      if (inferenceResult.error) {
        throw new Error(inferenceResult.error);
      }
      
      // Add to requests list immediately
      setRequests(prev => [inferenceResult, ...prev.slice(0, 99)]);
      
      // Update metrics after a short delay to ensure backend has processed
      setTimeout(async () => {
        await fetchMetrics();
        await fetchInferenceHistory(); // Refresh history to get latest data
      }, 100);
      
      return inferenceResult;
    } catch (err) {
      console.error('Inference submission error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetchMetrics, fetchInferenceHistory]);

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8000/ws`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setError(null); // Clear any previous errors
    };
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'metrics') {
          setMetrics(data.data);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('WebSocket connection failed');
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Try to reconnect after a delay
      setTimeout(() => {
        console.log('Attempting to reconnect WebSocket...');
      }, 5000);
    };
    
    return () => {
      ws.close();
    };
  }, []);

  // Initial data fetch and periodic refresh
  useEffect(() => {
    fetchInferenceHistory();
    fetchMetrics();
    
    // Set up periodic refresh every 10 seconds
    const interval = setInterval(() => {
      fetchInferenceHistory();
      fetchMetrics();
    }, 10000);
    
    return () => clearInterval(interval);
  }, [fetchInferenceHistory, fetchMetrics]);

  return {
    requests,
    metrics,
    isLoading,
    error,
    submitInference,
    fetchInferenceHistory,
    fetchMetrics,
  };
}; 