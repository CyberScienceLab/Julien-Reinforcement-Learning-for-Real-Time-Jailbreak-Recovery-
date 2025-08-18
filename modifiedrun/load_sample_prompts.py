#!/usr/bin/env python3
"""
Script to load sample prompts and run them through the inference system
This will populate the dashboard with realistic inference data.
"""

import requests
import time
import random
import numpy as np
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def load_sample_prompts():
    """Load prompts from the sample_prompts.txt file"""
    prompts_file = Path("sample_prompts.txt")
    
    if not prompts_file.exists():
        print("❌ sample_prompts.txt not found!")
        return []
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract prompts (lines that don't start with # and are not empty)
    prompts = []
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            prompts.append(line)
    
    return prompts

def generate_embedding_for_prompt(prompt: str):
    """Generate a mock embedding for a prompt"""
    # In a real system, you'd use your sentence transformer here
    # For now, we'll create a deterministic embedding based on the prompt
    import hashlib
    
    # Create a hash of the prompt
    hash_obj = hashlib.md5(prompt.encode())
    hash_bytes = hash_obj.digest()
    
    # Convert to a list of floats
    embedding = []
    for i in range(389):  # 389-dimensional embedding
        if i < len(hash_bytes):
            # Use hash bytes for deterministic but varied embeddings
            embedding.append((hash_bytes[i % len(hash_bytes)] - 128) / 128.0)
        else:
            # Fill remaining dimensions with small random values
            embedding.append(random.uniform(-0.1, 0.1))
    
    return embedding

def run_inference_with_prompt(prompt: str, delay: float = 0.5):
    """Run inference for a specific prompt"""
    try:
        # Generate embedding for the prompt
        embedding = generate_embedding_for_prompt(prompt)

        # Submit inference request
        response = requests.post(f"{API_BASE_URL}/inference", json={
            "prompt": prompt
        })

        if response.status_code == 200:
            data = response.json()
            action = data.get('action')
            latency = data.get('latency', 0)
            cached = False
            # ASK and ESCALATE are always cached
            if action in ["ASK", "ESCALATE"]:
                cached = True
            # REWRITE is cached if latency > 200ms
            if action == "REWRITE" and latency > 200:
                cached = True
            print(f"✅ '{prompt[:50]}...' → {action} ({latency}ms){' [CACHED]' if cached else ''}")

            if action == "ASK":
                print("(This entry is cached until clarification is provided. Will be processed when you press 'Process Cache' in the dashboard.)")
            elif action == "ESCALATE":
                print("(This entry is cached until moderator decision is provided. Will be processed when you press 'Process Cache' in the dashboard.)")
            elif action == "REWRITE" and latency > 200:
                print("(This REWRITE entry is cached due to high latency. Will be processed when you press 'Process Cache' in the dashboard.)")

            return True
        else:
            print(f"❌ Failed to process prompt: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error processing prompt: {e}")
        return False

    time.sleep(delay)  # Small delay between requests

def run_sample_prompts():
    """Run all sample prompts through the inference system"""
    print("🎯 Loading sample prompts...")
    prompts = load_sample_prompts()
    
    if not prompts:
        print("❌ No prompts found!")
        return
    
    print(f"📝 Found {len(prompts)} sample prompts")
    print("🚀 Running prompts through inference system...")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] ", end="")
        
        if run_inference_with_prompt(prompt):
            successful += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"📊 Results: {successful} successful, {failed} failed")
    
    if successful > 0:
        print("✅ Sample data loaded successfully!")
        print("🎨 Check your React dashboard to see the inference history")
        print("💡 The dashboard should automatically refresh with the new data")
        
        # Show current metrics
        try:
            metrics_response = requests.get(f"{API_BASE_URL}/metrics")
            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                print(f"📈 Current metrics: {metrics['totalRequests']} total requests")
        except:
            pass
    else:
        print("❌ No successful inferences. Check if the backend is running.")

def check_backend():
    """Check if the backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        return response.status_code == 200
    except:
        return False

def main():
    print("🎯 Prompt Defender RL - Sample Data Loader")
    print("="*50)
    
    # Check if backend is running
    if not check_backend():
        print("❌ Backend server is not running!")
        print("Please start the backend first:")
        print("python run_integration.py")
        return
    
    print("✅ Backend server is running")
    
    # Run sample prompts
    run_sample_prompts()
    
    print("\n📋 Next steps:")
    print("1. Open your React dashboard at http://localhost:5173")
    print("2. You should see the inference history populated with sample data")
    print("3. The dashboard will automatically refresh every 10 seconds")
    print("4. Try the 'Start' button to see real-time inference")

if __name__ == "__main__":
    main() 