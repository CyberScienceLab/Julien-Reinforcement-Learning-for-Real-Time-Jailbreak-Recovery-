#!/usr/bin/env python3
"""
Integration script for Prompt Defender RL
This script starts the Python backend and provides instructions for the React frontend.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn[standard]',  # Includes WebSocket support
        'onnxruntime',
        'numpy',
        'ollama',
        'pydantic',
        'websockets'  # For WebSocket support
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'uvicorn[standard]':
                import uvicorn
                # Check if websockets is available
                try:
                    import websockets
                except ImportError:
                    missing_packages.append('websockets')
            else:
                __import__(package.replace('[standard]', ''))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them with:")
        print("pip install uvicorn[standard] websockets")
        print("OR")
        print("pip install fastapi uvicorn onnxruntime numpy ollama pydantic websockets")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_model_file():
    """Check if the ONNX model file exists"""
    model_path = Path("modelF.onnx")
    if not model_path.exists():
        print("❌ Model file 'modelF.onnx' not found in current directory")
        print("Please ensure the model file is in the Pipeline directory")
        return False
    
    print("✅ Model file found")
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    print("📡 Backend will be available at: http://localhost:8000")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("\n" + "="*50)
    
    try:
        # Start uvicorn server with WebSocket support
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000",
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Backend server stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    
    return True

def print_frontend_instructions():
    """Print instructions for running the React frontend"""
    print("\n" + "="*50)
    print("🎨 REACT FRONTEND INSTRUCTIONS")
    print("="*50)
    print("⚠️  IMPORTANT: You need to run the frontend in a SEPARATE terminal!")
    print("\n1. Open a NEW terminal window/tab")
    print("2. Navigate to the React frontend directory:")
    print("   cd ../prompt-defender-rl")
    print("3. Install dependencies (if not already done):")
    print("   npm install")
    print("4. Start the React development server:")
    print("   npm run dev")
    print("5. Open your browser to: http://localhost:5173")
    print("\nThe React app will automatically connect to the Python backend!")
    print("="*50)

def main():
    print("🔒 Prompt Defender RL Integration")
    print("="*50)
    print("This script starts the Python backend server.")
    print("The React frontend needs to be started separately.")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check model file
    if not check_model_file():
        return
    
    print("\n📋 Backend endpoints that will be available:")
    print("• POST /predict - Basic prediction")
    print("• POST /inference - Full inference with metrics")
    print("• GET /inference/history - Get inference history")
    print("• GET /metrics - Get aggregated metrics")
    print("• WebSocket /ws - Real-time updates")
    
    print_frontend_instructions()
    
    print("\n🚀 Starting backend server now...")
    print("(Press Ctrl+C to stop the backend)")
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main() 