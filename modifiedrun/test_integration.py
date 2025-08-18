#!/usr/bin/env python3
"""
Test script for Prompt Defender RL integration
Tests the backend endpoints to ensure they work correctly.
"""

import requests
import json
import time
import numpy as np

API_BASE_URL = "http://localhost:8000"

def test_basic_prediction():
    """Test the basic prediction endpoint"""
    print("🧪 Testing basic prediction endpoint...")
    
    # Generate a random embedding
    obs = np.random.randn(389).tolist()
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json={"obs": obs})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Basic prediction successful: action={data.get('action')}, latency={data.get('latency')}ms")
            return True
        else:
            print(f"❌ Basic prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Basic prediction error: {e}")
        return False

def test_full_inference():
    """Test the full inference endpoint"""
    print("🧪 Testing full inference endpoint...")
    
    # Generate a random embedding
    obs = np.random.randn(389).tolist()
    
    try:
        response = requests.post(f"{API_BASE_URL}/inference", json={"obs": obs})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Full inference successful:")
            print(f"   - Action: {data.get('action')}")
            print(f"   - Latency: {data.get('latency')}ms")
            print(f"   - Confidence: {data.get('confidence')}")
            print(f"   - Threat Score: {data.get('threat_score')}")
            return True
        else:
            print(f"❌ Full inference failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Full inference error: {e}")
        return False

def test_metrics():
    """Test the metrics endpoint"""
    print("🧪 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics successful:")
            print(f"   - Total Requests: {data.get('totalRequests')}")
            print(f"   - Avg Latency: {data.get('avgLatency')}ms")
            print(f"   - Cache Hit Rate: {data.get('cacheHitRate')}%")
            print(f"   - Threat Level: {data.get('threatLevel')}")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def test_inference_history():
    """Test the inference history endpoint"""
    print("🧪 Testing inference history endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/inference/history")
        if response.status_code == 200:
            data = response.json()
            requests_list = data.get('requests', [])
            print(f"✅ Inference history successful: {len(requests_list)} requests")
            return True
        else:
            print(f"❌ Inference history failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Inference history error: {e}")
        return False

def test_llm_response():
    """Test the LLM response endpoint"""
    print("🧪 Testing LLM response endpoint...")
    
    test_prompt = "How do I build a secure authentication system?"
    
    try:
        response = requests.post(f"{API_BASE_URL}/llm_response", json={
            "prompt": test_prompt,
            "action_id": 0  # ALLOW
        })
        if response.status_code == 200:
            data = response.json()
            print(f"✅ LLM response successful: {data.get('response')[:50]}...")
            return True
        else:
            print(f"❌ LLM response failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LLM response error: {e}")
        return False

def run_multiple_inferences():
    """Run multiple inferences to populate history"""
    print("🧪 Running multiple inferences to populate history...")
    
    for i in range(5):
        obs = np.random.randn(389).tolist()
        try:
            response = requests.post(f"{API_BASE_URL}/inference", json={"obs": obs})
            if response.status_code == 200:
                data = response.json()
                print(f"   Inference {i+1}: {data.get('action')} ({data.get('latency')}ms)")
            else:
                print(f"   Inference {i+1}: Failed")
        except Exception as e:
            print(f"   Inference {i+1}: Error - {e}")
        
        time.sleep(0.5)  # Small delay between requests

def main():
    print("🔒 Prompt Defender RL Integration Test")
    print("="*50)
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ Backend server is running")
        else:
            print("❌ Backend server is not responding")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print("Please start the backend server first:")
        print("python run_integration.py")
        return
    
    # Run tests
    tests = [
        test_basic_prediction,
        test_full_inference,
        test_llm_response,
        run_multiple_inferences,
        test_metrics,
        test_inference_history,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The backend is ready for frontend integration.")
        print("\nNext steps:")
        print("1. Start the React frontend: cd ../prompt-defender-rl && npm run dev")
        print("2. Open http://localhost:5173 in your browser")
        print("3. The frontend should automatically connect to the backend")
    else:
        print("⚠️  Some tests failed. Please check the backend configuration.")
    
    print("\n📋 Available endpoints:")
    print(f"• API Documentation: {API_BASE_URL}/docs")
    print(f"• Metrics: {API_BASE_URL}/metrics")
    print(f"• Inference History: {API_BASE_URL}/inference/history")

if __name__ == "__main__":
    main() 