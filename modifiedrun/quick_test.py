#!/usr/bin/env python3
"""
Quick test to see if backend has data and frontend can access it
"""

import requests
import json

def test_backend_data():
    """Test if backend has any data"""
    print("🔍 Testing backend data...")
    
    try:
        # Test 1: Check if backend is running
        response = requests.get("http://localhost:8000/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Backend is running")
            print(f"📊 Total requests: {data.get('totalRequests', 0)}")
            print(f"📊 Threat level: {data.get('threatLevel', 'UNKNOWN')}")
        else:
            print(f"❌ Backend not responding: {response.status_code}")
            return False
            
        # Test 2: Check inference history
        response = requests.get("http://localhost:8000/inference/history")
        if response.status_code == 200:
            data = response.json()
            requests_list = data.get('requests', [])
            print(f"📜 History has {len(requests_list)} requests")
            
            if len(requests_list) == 0:
                print("⚠️  No data in backend - this is why frontend shows empty!")
                return False
            else:
                print("✅ Backend has data!")
                return True
        else:
            print(f"❌ History endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def add_test_data():
    """Add some test data to the backend"""
    print("\n🧪 Adding test data...")
    
    test_embedding = [0.1] * 389
    
    for i in range(5):
        try:
            response = requests.post(
                "http://localhost:8000/inference",
                json={"obs": test_embedding},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Added request {i+1}: {result['action']}")
            else:
                print(f"❌ Failed to add request {i+1}")
                
        except Exception as e:
            print(f"❌ Error adding request {i+1}: {e}")

def main():
    print("🔍 Quick Backend/Frontend Connection Test")
    print("="*50)
    
    # Check if backend has data
    has_data = test_backend_data()
    
    if not has_data:
        print("\n📝 Adding test data to backend...")
        add_test_data()
        
        print("\n🔄 Checking again...")
        test_backend_data()
    
    print("\n📋 Next steps:")
    print("1. If backend has data, refresh your React app")
    print("2. Check browser console (F12) for fetch logs")
    print("3. The frontend should now show the data!")

if __name__ == "__main__":
    main() 