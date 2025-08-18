#!/bin/bash
"""
Frontend startup script for Prompt Defender RL
This script starts the React frontend development server.
"""

echo "🎨 Prompt Defender RL Frontend"
echo "=============================="
echo "Starting React development server..."
echo ""

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found!"
    echo "Please run this script from the prompt-defender-rl directory"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

echo "🚀 Starting React development server..."
echo "📡 Frontend will be available at: http://localhost:5173"
echo "🔗 Backend should be running at: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the frontend server"
echo ""

# Start the development server
npm run dev 