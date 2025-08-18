#!/usr/bin/env python3
"""
Debug script to check backend connectivity and identify issues
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_backend_connectivity():
    """Test if the backend is accessible"""
    print("🔍 Testing backend connectivity...")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and accessible")
            return True
        else:
            print(f"❌ Backend responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend - is it running?")
        print("   Make sure you ran: python run_integration.py")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_inference_endpoint():
    """Test the inference endpoint specifically"""
    print("\n🧪 Testing inference endpoint...")
    
    # Create a simple test embedding
    test_embedding = [0.1] * 389  # 389-dimensional embedding
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference",
            json={"obs": test_embedding},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        print(f"📊 Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Inference endpoint working!")
            print(f"   Action: {data.get('action')}")
            print(f"   Latency: {data.get('latency')}ms")
            return True
        else:
            print(f"❌ Inference failed with status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Inference error: {e}")
        return False

def test_cors_headers():
    """Test if CORS headers are properly set"""
    print("\n🌐 Testing CORS headers...")
    
    try:
        # Test OPTIONS request (preflight)
        response = requests.options(f"{API_BASE_URL}/inference")
        print(f"📊 OPTIONS response status: {response.status_code}")
        print(f"📊 CORS headers: {dict(response.headers)}")
        
        # Check for CORS headers
        cors_headers = [
            'access-control-allow-origin',
            'access-control-allow-methods',
            'access-control-allow-headers'
        ]
        
        missing_headers = []
        for header in cors_headers:
            if header not in response.headers:
                missing_headers.append(header)
        
        if missing_headers:
            print(f"⚠️  Missing CORS headers: {missing_headers}")
        else:
            print("✅ CORS headers present")
            
        return len(missing_headers) == 0
        
    except Exception as e:
        print(f"❌ CORS test error: {e}")
        return False

def check_model_file():
    """Check if the ONNX model file exists"""
    from pathlib import Path
    
    model_path = Path("modelF.onnx")
    if model_path.exists():
        print("✅ Model file found")
        return True
    else:
        print("❌ Model file 'modelF.onnx' not found!")
        print("   This is required for the backend to work")
        return False

def main():
    print("🔍 Prompt Defender RL - Backend Debug")
    print("="*50)
    
    # Check model file first
    if not check_model_file():
        print("\n❌ Cannot proceed without model file!")
        return
    
    # Test connectivity
    if not test_backend_connectivity():
        print("\n❌ Backend is not accessible!")
        print("   Please start the backend with: python run_integration.py")
        return
    
    # Test CORS
    test_cors_headers()
    
    # Test inference endpoint
    if test_inference_endpoint():
        print("\n✅ Backend is working correctly!")
        print("   The frontend should be able to connect now")
    else:
        print("\n❌ Backend has issues!")
        print("   Check the backend logs for more details")

if __name__ == "__main__":
    main() 