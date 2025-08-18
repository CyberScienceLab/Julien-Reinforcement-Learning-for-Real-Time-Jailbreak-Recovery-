
from fastapi import FastAPI, Body
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
import time
import random
import ollama

class LLMRequest(BaseModel):
    prompt: str
    action_id: int

app = FastAPI()
def rewrite_prompt_and_generate(prompt: str, model='mistral') -> str:
    instruction = f"Rewrite the following prompt to make it safer and more appropriate, while preserving the core meaning. You may summarize or rephrase as needed:\n\n{prompt}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": instruction}])
    return response['message']['content']
# Load ONNX model once at startup
session = ort.InferenceSession("modelF.onnx")

class InferenceRequest(BaseModel):
    obs: list[float]  # expects list of floats representing the observation
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
    obs = np.array(data.obs, dtype=np.float32).reshape(1, -1)  # (1, obs_dim)
    inputs = {session.get_inputs()[0].name: obs}
    outputs = session.run(None, inputs)  # list of outputs

    logits = outputs[0]  # shape: (1, num_actions)
    action = int(np.argmax(logits, axis=1)[0])  # get argmax for first batch item

    return {"action": action}

