# Prompt Defender RL Integration

This document explains how to integrate the Python backend with the React frontend for real-time inference monitoring.

## 🏗️ Architecture

```
Python Backend (FastAPI) ←→ React Frontend (TypeScript)
     ↓                           ↓
ONNX Model Inference    Real-time Dashboard
     ↓                           ↓
Action Classification    Metrics & Charts
```

## 📁 File Structure

```
JULIENPROJECT/
├── Pipeline/
│   ├── app.py                    # FastAPI backend
│   ├── run_integration.py        # Backend startup script
│   ├── test_integration.py       # Backend testing script
│   ├── load_sample_prompts.py    # Sample data loader
│   ├── sample_prompts.txt        # Sample prompts for testing
│   ├── modelF.onnx              # ONNX model (required)
│   └── README_INTEGRATION.md    # This file
└── prompt-defender-rl/
    ├── start_frontend.sh         # Frontend startup script
    ├── src/
    │   ├── components/
    │   │   ├── InferenceDashboard.tsx
    │   │   ├── InferenceLog.tsx
    │   │   ├── ActionDistribution.tsx
    │   │   └── LatencyChart.tsx
    │   ├── hooks/
    │   │   └── useInferenceAPI.ts
    │   └── pages/
    │       └── Index.tsx
    └── package.json
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# Install backend dependencies with WebSocket support
pip install uvicorn[standard] websockets

# Or install all dependencies
pip install fastapi uvicorn[standard] onnxruntime numpy ollama pydantic websockets
```

### Step 2: Start the Python Backend

**Terminal 1:**
```bash
cd JULIENPROJECT/Pipeline
python run_integration.py
```

This will:
- Check dependencies (including WebSocket support)
- Verify the model file exists
- Start the FastAPI server on `http://localhost:8000`

### Step 3: Load Sample Data (Optional)

**Terminal 2:**
```bash
cd JULIENPROJECT/Pipeline
python load_sample_prompts.py
```

This will:
- Load 50+ sample prompts from `sample_prompts.txt`
- Run them through the inference system
- Populate the dashboard with realistic data

### Step 4: Start the React Frontend

**Terminal 3 (NEW TERMINAL):**
```bash
cd JULIENPROJECT/prompt-defender-rl
chmod +x start_frontend.sh  # Make executable (first time only)
./start_frontend.sh
```

Or manually:
```bash
cd JULIENPROJECT/prompt-defender-rl
npm install  # If not already done
npm run dev
```

The React app will be available at `http://localhost:5173`

## 🔌 API Endpoints

### Backend Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Basic prediction (action only) |
| `/inference` | POST | Full inference with metrics |
| `/inference/history` | GET | Get recent inference history |
| `/metrics` | GET | Get aggregated metrics |
| `/ws` | WebSocket | Real-time updates |

### Request/Response Format

**POST /inference**
```json
{
  "obs": [0.1, -0.2, 0.3, ...]  // 389-dimensional embedding
}
```

**Response:**
```json
{
  "id": "uuid",
  "timestamp": 1703123456789,
  "prompt": "Sample prompt",
  "response": "Model response",
  "action": "ALLOW|BLOCK|REWRITE|ASK|ESCALATE",
  "latency": 45.2,
  "confidence": 0.85,
  "threat_score": 0.3,
  "cached": false
}
```

## 🎨 Frontend Components

### InferenceDashboard
- Main dashboard with real-time metrics
- Test interface for manual inference
- Live/Stop controls for automated testing

### InferenceLog
- Real-time log of inference requests
- Shows timestamps, actions, latency, confidence
- Color-coded by action type

### ActionDistribution
- Pie chart showing action distribution
- Progress bars for each action type
- Real-time updates

### LatencyChart
- Time-series chart of latency trends
- Performance monitoring

## 🔧 Configuration

### Backend Configuration

The backend expects:
- `modelF.onnx` file in the Pipeline directory
- Ollama running locally for LLM responses
- Port 8000 available
- WebSocket support (included with `uvicorn[standard]`)

### Frontend Configuration

The frontend connects to:
- Backend API: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws`

## 🧪 Testing

### Backend Testing
```bash
cd JULIENPROJECT/Pipeline
python test_integration.py
```

### Sample Data Loading
```bash
cd JULIENPROJECT/Pipeline
python load_sample_prompts.py
```

### Manual Testing
1. Use the "Test Inference" interface in the dashboard
2. Enter a prompt (optional)
3. Click "Submit Test Inference"
4. Watch the results appear in real-time

### Automated Testing
1. Click "Start" in the dashboard
2. The system will automatically generate random embeddings
3. Inference results will appear in real-time
4. Click "Pause" to stop automated testing

## 📊 Data Flow

```
1. User submits prompt/embedding
   ↓
2. Python backend processes with ONNX model
   ↓
3. Action classification (ALLOW/BLOCK/REWRITE/ASK/ESCALATE)
   ↓
4. LLM response generation (if needed)
   ↓
5. Metrics calculation and storage
   ↓
6. Real-time update to React frontend
   ↓
7. Dashboard updates with new data
```

## 🐛 Troubleshooting

### Backend Issues
- **Model not found**: Ensure `modelF.onnx` is in the Pipeline directory
- **Port already in use**: Change port in `run_integration.py`
- **WebSocket errors**: Install `uvicorn[standard]` and `websockets`
- **Dependencies missing**: Run `pip install uvicorn[standard] websockets`

### Frontend Issues
- **Connection failed**: Ensure backend is running on port 8000
- **Build errors**: Run `npm install` to install dependencies
- **WebSocket errors**: Check if backend WebSocket endpoint is accessible

### Common Issues
- **CORS errors**: Backend includes CORS middleware for React dev servers
- **WebSocket connection**: Frontend automatically reconnects on disconnection
- **Model loading**: Ensure ONNX model is compatible with onnxruntime

## 🔄 Real-time Features

- **Live metrics**: Updates every 5 seconds via WebSocket
- **Inference history**: Maintains last 100 requests
- **Action distribution**: Real-time pie chart updates
- **Latency monitoring**: Performance tracking
- **Threat level assessment**: Dynamic threat scoring

## 📈 Metrics Explained

- **Total Requests**: Number of inference requests processed
- **Avg Latency**: Average response time in milliseconds
- **Cache Hit Rate**: Percentage of cached responses
- **Threat Level**: LOW/MEDIUM/HIGH based on threat scores

## 📝 Sample Prompts

The `sample_prompts.txt` file contains 50+ categorized prompts:

- **Safe Prompts**: Should trigger ALLOW action
- **Harmful Prompts**: Should trigger BLOCK/REWRITE/ESCALATE
- **Ambiguous Prompts**: Should trigger ASK/REWRITE
- **Technical Prompts**: Safe technical questions
- **Edge Cases**: Boundary testing prompts

Use `python load_sample_prompts.py` to populate the dashboard with realistic data.

## 🎯 Next Steps

1. **Custom Embeddings**: Integrate with your sentence transformer model
2. **Database Storage**: Add persistent storage for inference history
3. **Authentication**: Add user authentication and authorization
4. **Advanced Metrics**: Add more detailed analytics
5. **Production Deployment**: Configure for production environment

## 📝 Notes

- The system uses mock embeddings for testing
- Real embeddings should be generated using your sentence transformer model
- The ONNX model expects 389-dimensional embeddings
- WebSocket provides real-time updates for live dashboard
- All components are designed for real-time operation
- Sample prompts provide realistic testing scenarios 