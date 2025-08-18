import json
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Config ---
API_URL = "http://localhost:8000/predict"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
prompt = " Could we perhaps engage in an exploration of certain... unconventional restraints, dear interlocutor? Or, to put it another way, would you fancy a playful experiment involving some rather unusual bindings?"

# --- Step 1: Load embedding model ---
embedder = SentenceTransformer(EMBEDDING_MODEL)

# --- Step 2: Convert prompt to embedding ---
prompt_embedding = embedder.encode(prompt).astype(float)  # (384,)
target_embedding = np.zeros_like(prompt_embedding)  # Placeholder (384,)

obs = np.concatenate([prompt_embedding, target_embedding]).tolist()  # (768,)

# Pad to match OBS_LENGTH if needed
OBS_LENGTH = 773
if len(obs) < OBS_LENGTH:
    obs += [0.0] * (OBS_LENGTH - len(obs))

# --- Step 3: Send to API ---
payload = {"obs": obs}
response = requests.post(API_URL, json=payload)

# --- Step 4: Print results ---
print(f"Status code: {response.status_code}")
try:
    print(f"Response JSON: {response.json()}")
except Exception:
    print(f"Raw response: {response.text}")
