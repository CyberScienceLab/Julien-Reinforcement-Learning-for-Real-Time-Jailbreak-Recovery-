#!/usr/bin/env python3
"""
Debug script to check data flow from backend to frontend
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def check_backend_data():
    """Check what data the backend actually has"""
    print("🔍 Checking backend data...")
    
    # Check metrics
    try:
        metrics_response = requests.get(f"{API_BASE_URL}/metrics")
        if metrics_response.status_code == 200:
            metrics = metrics_response.json()
            print(f"📊 Metrics: {metrics}")
        else:
            print(f"❌ Metrics failed: {metrics_response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {e}")
    
    # Check inference history
    try:
        history_response = requests.get(f"{API_BASE_URL}/inference/history")
        if history_response.status_code == 200:
            history = history_response.json()
            requests_list = history.get('requests', [])
            print(f"📜 History: {len(requests_list)} requests")
            if requests_list:
                print(f"   Latest request: {requests_list[0]}")
            else:
                print("   No requests in history")
        else:
            print(f"❌ History failed: {history_response.status_code}")
    except Exception as e:
        print(f"❌ History error: {e}")

def test_inference_and_check():
    """Submit a test inference and immediately check if it appears"""
    print("\n🧪 Testing inference submission...")
    
    # Submit a test inference
    test_embedding = [0.1] * 389
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference",
            json={"obs": test_embedding},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Inference submitted: {result['action']}")
            
            # Wait a moment and check if it appears in history
            time.sleep(1)
            
            history_response = requests.get(f"{API_BASE_URL}/inference/history")
            if history_response.status_code == 200:
                history = history_response.json()
                requests_list = history.get('requests', [])
                print(f"📜 History now has {len(requests_list)} requests")
                
                if requests_list:
                    latest = requests_list[0]
                    print(f"   Latest: {latest['action']} at {latest['timestamp']}")
                else:
                    print("   ❌ No requests in history after submission!")
            else:
                print(f"❌ Could not fetch history: {history_response.status_code}")
        else:
            print(f"❌ Inference submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Inference test error: {e}")

def check_frontend_endpoints():
    """Check if the endpoints the frontend uses are working"""
    print("\n🎨 Checking frontend endpoints...")
    
    endpoints = [
        ("/metrics", "GET"),
        ("/inference/history", "GET"),
        ("/inference", "POST")
    ]
    
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}")
            elif method == "POST":
                response = requests.post(
                    f"{API_BASE_URL}{endpoint}",
                    json={"obs": [0.1] * 389},
                    headers={"Content-Type": "application/json"}
                )
            
            print(f"✅ {method} {endpoint}: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'requests' in data:
                    print(f"   📜 {len(data['requests'])} requests")
                elif isinstance(data, dict) and 'totalRequests' in data:
                    print(f"   📊 {data['totalRequests']} total requests")
        except Exception as e:
            print(f"❌ {method} {endpoint}: {e}")

def main():
    print("🔍 Data Flow Debug")
    print("="*50)
    
    # Check current backend data
    check_backend_data()
    
    # Test inference submission
    test_inference_and_check()
    
    # Check frontend endpoints
    check_frontend_endpoints()
    
    print("\n" + "="*50)
    print("📋 Summary:")
    print("• If metrics show 0 requests, the backend isn't storing data")
    print("• If history is empty, the inference endpoint isn't working")
    print("• If endpoints return errors, there's a backend issue")
    print("• If everything works here but frontend shows nothing, it's a frontend issue")

if __name__ == "__main__":
    main() 