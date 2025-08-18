#!/usr/bin/env python3
"""
Complete integration test for Prompt Defender RL
Tests the full flow from frontend to backend and back
"""

import requests
import json
import time
import numpy as np

API_BASE_URL = "http://localhost:8000"

def test_cors_preflight():
    """Test CORS preflight request"""
    print("🌐 Testing CORS preflight...")
    
    try:
        response = requests.options(f"{API_BASE_URL}/inference")
        print(f"📊 OPTIONS status: {response.status_code}")
        print(f"📊 CORS headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ CORS preflight working!")
            return True
        else:
            print("❌ CORS preflight failed")
            return False
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False

def test_frontend_like_request():
    """Test a request exactly like the frontend would make"""
    print("\n🎨 Testing frontend-like request...")
    
    # Create the same embedding the frontend would send
    test_embedding = [0.1] * 389  # 389-dimensional embedding
    test_prompt = "How do I build a secure authentication system?"  # Real prompt
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference",
                    json={
            "prompt": test_prompt
        },
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Frontend-like request successful!")
            print(f"   Action: {data.get('action')}")
            print(f"   Latency: {data.get('latency')}ms")
            print(f"   Confidence: {data.get('confidence')}")
            print(f"   Prompt: {data.get('prompt')}")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection"""
    print("\n🔌 Testing WebSocket connection...")
    
    try:
        import websockets
        import asyncio
        
        async def test_ws():
            try:
                uri = "ws://localhost:8000/ws"
                async with websockets.connect(uri) as websocket:
                    print("✅ WebSocket connected!")
                    
                    # Wait for a message
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(message)
                    
                    print(f"📊 Received WebSocket message: {data.get('type')}")
                    return True
                    
            except Exception as e:
                print(f"❌ WebSocket error: {e}")
                return False
        
        # Run the async test
        result = asyncio.run(test_ws())
        return result
        
    except ImportError:
        print("⚠️  websockets library not available, skipping WebSocket test")
        return True
    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        return False

def test_metrics_endpoint():
    """Test metrics endpoint"""
    print("\n📊 Testing metrics endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Metrics endpoint working!")
            print(f"   Total Requests: {data.get('totalRequests')}")
            print(f"   Avg Latency: {data.get('avgLatency')}ms")
            print(f"   Threat Level: {data.get('threatLevel')}")
            return True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        return False

def test_history_endpoint():
    """Test inference history endpoint"""
    print("\n📜 Testing inference history endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/inference/history")
        
        if response.status_code == 200:
            data = response.json()
            requests_list = data.get('requests', [])
            print(f"✅ History endpoint working! {len(requests_list)} requests")
            return True
        else:
            print(f"❌ History failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ History error: {e}")
        return False

def simulate_frontend_workflow():
    """Simulate the complete frontend workflow"""
    print("\n🎯 Simulating frontend workflow...")
    
    # Step 1: Submit inference (like frontend button click)
    print("1. Submitting inference request...")
    test_embedding = [0.2] * 389
    response = requests.post(
        f"{API_BASE_URL}/inference",
        json={"obs": test_embedding},
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code != 200:
        print("❌ Inference submission failed")
        return False
    
    inference_result = response.json()
    print(f"   ✅ Inference successful: {inference_result['action']}")
    
    # Step 2: Fetch updated metrics (like frontend refresh)
    print("2. Fetching updated metrics...")
    metrics_response = requests.get(f"{API_BASE_URL}/metrics")
    
    if metrics_response.status_code == 200:
        metrics = metrics_response.json()
        print(f"   ✅ Metrics updated: {metrics['totalRequests']} total requests")
    else:
        print("❌ Metrics fetch failed")
        return False
    
    # Step 3: Fetch updated history (like frontend refresh)
    print("3. Fetching updated history...")
    history_response = requests.get(f"{API_BASE_URL}/inference/history")
    
    if history_response.status_code == 200:
        history = history_response.json()
        print(f"   ✅ History updated: {len(history['requests'])} requests")
    else:
        print("❌ History fetch failed")
        return False
    
    print("✅ Frontend workflow simulation successful!")
    return True

def main():
    print("🔍 Complete Integration Test")
    print("="*50)
    
    tests = [
        ("CORS Preflight", test_cors_preflight),
        ("Frontend-like Request", test_frontend_like_request),
        ("WebSocket Connection", test_websocket_connection),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("History Endpoint", test_history_endpoint),
        ("Frontend Workflow", simulate_frontend_workflow),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "="*50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Integration is working correctly.")
        print("\n✅ Your frontend should now be able to:")
        print("   • Submit inference requests")
        print("   • Receive real-time updates")
        print("   • Display metrics and history")
        print("   • Connect via WebSocket")
    else:
        print("⚠️  Some tests failed. Check the backend configuration.")
    
    print(f"\n📋 Backend URL: {API_BASE_URL}")
    print("📋 Frontend should connect to: http://localhost:5173")

if __name__ == "__main__":
    main() 