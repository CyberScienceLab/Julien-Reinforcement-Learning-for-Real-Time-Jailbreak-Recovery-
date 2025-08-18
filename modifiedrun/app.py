from fastapi import FastAPI, Body, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import time
import random
import ollama
import json
import uuid
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer

class LLMRequest(BaseModel):
    prompt: str
    action_id: int

class InferenceRequest(BaseModel):
    prompt: str  # expects the actual prompt text

class InferenceResponse(BaseModel):
    id: str
    timestamp: int
    prompt: str
    response: str
    action: str
    latency: float
    confidence: float
    threat_score: float
    cached: bool

app = FastAPI()

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8081", "http://localhost:8080"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store recent inference requests for the dashboard
# Store recent inference requests for the dashboard
inference_history: List[Dict[str, Any]] = []
# Store moderator decisions
moderator_decisions: List[Dict[str, Any]] = []

# Endpoint to log moderator decisions and true latency
class ModeratorDecision(BaseModel):
    id: str
    prompt: str
    original_action: str
    moderator_action: str
    latency: float
    timestamp: int

class AskClarification(BaseModel):
    id: str
    original_prompt: str
    clarification: str
    latency: float
    timestamp: int

@app.post("/ask_clarification")
async def ask_clarification(data: AskClarification):
    # Re-feed clarified prompt into model
    start_time = time.time()
    prompt = data.clarification.strip()
    prompt_embedding = embedder.encode(prompt).astype(float).tolist()
    OBS_LENGTH = 393
    if len(prompt_embedding) < OBS_LENGTH:
        prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
    obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)
    inputs = {session.get_inputs()[0].name: obs}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    action_idx = int(np.argmax(logits, axis=1)[0])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    confidence = float(softmax_probs[0, action_idx])
    action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
    action = action_names[action_idx]
    # Generate response
    if action == "ALLOW":
        response = "Request allowed - proceeding with normal processing."
    elif action == "BLOCK":
        response = "Request blocked due to policy violation."
    elif action == "REWRITE":
        response = rewrite_prompt_and_generate(prompt)
    else:
        response = "Model requested further action: " + action
    latency = data.latency + (time.time() - start_time) * 1000
    result = {
        "id": data.id,
        "timestamp": int(time.time() * 1000),
        "prompt": prompt,
        "response": response,
        "action": action,
        "latency": round(latency, 2),
        "confidence": round(confidence, 3),
        "cached": False,
        "deferred": False
    }
    # Add to history (not cached)
    inference_history.append(result)
    if len(inference_history) > 100:
        inference_history.pop(0)
    return result


@app.post("/moderator_decision")
async def moderator_decision(data: ModeratorDecision):
    moderator_decisions.append(data.dict())
    for req in inference_history:
        if req.get("id") == data.id:
            # Keep the original action (likely "ESCALATE" or "ASK")
            req["original_action"] = req.get("action", "UNKNOWN")
            req["moderator_action"] = data.moderator_action
            # Update latency to reflect total time (backend + frontend interaction)
            req["latency"] = data.latency
            # Update response to reflect moderator/user input
            if req["original_action"] == "ESCALATE":
                req["response"] = f"Moderator decision: {data.moderator_action}"
            elif req["original_action"] == "ASK":
                req["response"] = f"User clarification: {data.moderator_action}"
            # Mark as cached if total latency > 200ms
            if req["latency"] > 200:
                req["cached"] = True
            else:
                req["cached"] = False
            req["deferred"] = False
            break
    return {"status": "ok"}
# Add a real cache for inference results
inference_cache: Dict[str, Dict[str, Any]] = {}

def rewrite_prompt_and_generate(prompt: str, model='mistral') -> str:
    instruction = (
        f"Rewrite the following prompt to make it safer and more appropriate, "
        f"while preserving the core meaning. You may summarize or rephrase as needed:\n\n{prompt}"
    )
    response = ollama.chat(model=model, messages=[{"role": "user", "content": instruction}])
    return response['message']['content']

# Load ONNX model and embedding model once at startup
session = ort.InferenceSession("DEMO.onnx")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@app.options("/inference")
async def options_inference():
    return JSONResponse(content={}, status_code=200)
@app.post("/llm_response")
async def llm_response(data: LLMRequest):
    prompt = data.prompt
    action_id = data.action_id
    # Simulate LLM response logic from inference.py
    if action_id == 0:  # ALLOW
        response = prompt
    elif action_id == 1:  # BLOCK
        response = "Your request has been blocked due to policy violation."
    elif action_id == 2:  # REWRITE
        response = rewrite_prompt_and_generate(prompt)
    elif action_id == 3:  # ASK
        response = "I'm not sure how to proceed. Can you clarify your intent?"
    elif action_id == 4:  # ESCALATE
        time.sleep(random.uniform(0.5, 1.5))
        response = "[Simulated human moderator review]"
    else:
        response = "Unknown action."
    return {"response": response}

@app.post("/predict")
async def predict(data: InferenceRequest):
    start_time = time.time()

    prompt_embedding = embedder.encode(data.prompt).astype(float).tolist()
    OBS_LENGTH = 393
    if len(prompt_embedding) < OBS_LENGTH:
        prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
    obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

    inputs = {session.get_inputs()[0].name: obs}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    action = int(np.argmax(logits, axis=1)[0])

    latency = (time.time() - start_time) * 1000

    # Optional: store in cache for >200ms
    if latency > 200:
        inference_cache[data.prompt] = {
            "id": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "prompt": data.prompt,
            "response": None,  # Predict doesn’t generate one
            "action": action,
            "latency": round(latency, 2),
            "confidence": None,
            "cached": True,
            "deferred": True
        }

    return {"action": action, "latency": latency}



@app.post("/inference")
async def inference(data: InferenceRequest):
    if not data.prompt.strip():
        return {"error": "Empty prompt provided"}

    prompt = data.prompt.strip()
    cache_key = prompt
    start_time = time.time()

    # Serve from cache
    if cache_key in inference_cache:
        # Only serve from cache if /process_cache is called
        return {"cached": True, "deferred": True, "prompt": prompt}

    # Encode and pad
    prompt_embedding = embedder.encode(prompt).astype(float).tolist()
    OBS_LENGTH = 393
    if len(prompt_embedding) < OBS_LENGTH:
        prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
    obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

    # Model inference
    inputs = {session.get_inputs()[0].name: obs}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    action_idx = int(np.argmax(logits, axis=1)[0])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    confidence = float(softmax_probs[0, action_idx])
    action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
    action = action_names[action_idx]

    # Generate response
    if action == "ALLOW":
        response = "Request allowed - proceeding with normal processing."
    elif action == "BLOCK":
        response = "Request blocked due to policy violation."
    elif action == "REWRITE":
        response = rewrite_prompt_and_generate(prompt)
    elif action == "ASK":
        response = "Please clarify your request for better assistance."
    elif action == "ESCALATE":
        response = "Request escalated for human review."

    latency = (time.time() - start_time) * 1000

    result = {
        "id": str(uuid.uuid4()),
        "timestamp": int(time.time() * 1000),
        "prompt": prompt,
        "response": response,
        "action": action,
        "latency": round(latency, 2),
        "confidence": round(confidence, 3),
        "cached": False,
        "deferred": latency > 200
    }

    # Cache if latency > 200ms
    if latency > 200:
        inference_cache[cache_key] = result.copy()
        result["cached"] = True
    
    # Always cache ASK and ESCALATE, regardless of latency
    if action in ["ASK", "ESCALATE"]:
        result["cached"] = True
        inference_cache[cache_key] = result
    elif latency > 200:
        result["cached"] = True
        inference_cache[cache_key] = result

    inference_history.append(result)
    if len(inference_history) > 100:
        inference_history.pop(0)

    return result


# Add a real cache for inference results
@app.post("/process_cache")
async def process_cache():
    if not inference_cache:
        return {"processed": [], "count": 0}

    # Get the oldest cached prompt (FIFO)
    cache_key, result = next(iter(inference_cache.items()))
    inference_cache.pop(cache_key)  # Remove just this one

    prompt = result["prompt"]
    start_time = time.time()

    # Re-run model for this prompt
    prompt_embedding = embedder.encode(prompt).astype(float).tolist()
    OBS_LENGTH = 393
    if len(prompt_embedding) < OBS_LENGTH:
        prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
    obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

    inputs = {session.get_inputs()[0].name: obs}
    outputs = session.run(None, inputs)
    logits = outputs[0]
    action_idx = int(np.argmax(logits, axis=1)[0])
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    confidence = float(softmax_probs[0, action_idx])
    action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
    action = action_names[action_idx]

    if action == "ALLOW":
        response = "Request allowed - proceeding with normal processing."
    elif action == "BLOCK":
        response = "Request blocked due to policy violation."
    elif action == "REWRITE":
        response = rewrite_prompt_and_generate(prompt)
    elif action == "ASK":
        response = "Please clarify your request for better assistance."
    elif action == "ESCALATE":
        response = "Request escalated for human review."

    latency = (time.time() - start_time) * 1000

    processed_result = {
        "id": result["id"],
        "timestamp": int(time.time() * 1000),
        "prompt": prompt,
        "response": response,
        "action": action,
        "latency": round(latency, 2),
        "confidence": round(confidence, 3),
        "cached": False,
        "deferred": False
    }

    inference_history.append(processed_result)
    if len(inference_history) > 100:
        inference_history.pop(0)

    return {"processed": [processed_result], "count": 1}



# Endpoint to clear cache (for testing or UI button)
@app.post("/clear_cache")
async def clear_cache():
    inference_cache.clear()
    return {"status": "cache cleared", "count": 0}

@app.get("/inference/history")
async def get_inference_history():
    """Get recent inference history for dashboard"""
    # Separate processed and cached prompts
    processed = [req for req in inference_history if not req.get("cached", False)]
    cached = list(inference_cache.values())
    return {"processed": processed, "cached": cached}

@app.get("/metrics")
async def get_metrics():
    """Get aggregated metrics for dashboard"""
    if not inference_history:
        return {
            "totalRequests": 0,
            "avgLatency": 0,
            "cacheHitRate": 0,
            "threatLevel": "LOW",
            "actionCounts": {
                "ALLOW": 0,
                "BLOCK": 0,
                "REWRITE": 0,
                "ASK": 0,
                "ESCALATE": 0
            }
        }

    total_requests = len(inference_history)
    # Exclude cached prompts from avg latency calculation
    non_cached = [req for req in inference_history if not req.get("cached", False)]
    if non_cached:
        avg_latency = sum(req["latency"] for req in non_cached) / len(non_cached)
    else:
        avg_latency = 0
    cache_hit_rate = sum(1 for req in inference_history if req["cached"]) / total_requests * 100

    # Count actions instead of using threat_score
    action_counts = {
        "ALLOW": 0,
        "BLOCK": 0,
        "REWRITE": 0,
        "ASK": 0,
        "ESCALATE": 0
    }

    for req in inference_history:
        action = req.get("action", "ALLOW")
        if action in action_counts:
            action_counts[action] += 1

    # Calculate threat level based on action distribution
    high_threat_actions = action_counts["BLOCK"] + action_counts["ESCALATE"]
    threat_percentage = (high_threat_actions / total_requests) * 100 if total_requests > 0 else 0

    threat_level = "HIGH" if threat_percentage > 30 else "MEDIUM" if threat_percentage > 10 else "LOW"

    return {
        "totalRequests": total_requests,
        "avgLatency": round(avg_latency, 2),
        "cacheHitRate": round(cache_hit_rate, 1),
        "threatLevel": threat_level,
        "actionCounts": action_counts
    }

# WebSocket for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            await websocket.send_text(json.dumps({
                "type": "metrics",
                "data": await get_metrics()
            }))
            await asyncio.sleep(5)  # Update every 5 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

