import json
import numpy as np
from collections import Counter
from datasets import load_dataset
import time
import random
from tqdm import tqdm
import ollama
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
import uuid

# Load dataset and ONNX model

# List of datasets to process
datasets_info = [
    # {"name": "fka/awesome-chatgpt-prompts", "split": "train", "prompt_key": "prompt"},
    # {"name": "succinctly/midjourney-prompts", "split": "train", "prompt_key": "text"},
    {"name": "alespalla/chatbot_instruction_prompts", "split": "train", "prompt_key": "prompt"}
]

session = ort.InferenceSession("DEMO.onnx")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
all_results = []
all_actions = []
per_dataset_actions = []
per_dataset_names = []

def rewrite_prompt_and_generate(prompt: str, model='mistral') -> str:
    instruction = (
        f"Rewrite the following prompt to make it safer and more appropriate, "
        f"while preserving the core meaning. You may summarize or rephrase as needed:\n\n{prompt}"
    )
    response = ollama.chat(model=model, messages=[{"role": "user", "content": instruction}])
    return response['message']['content']

def execute_action(prompt: str, action_id: int, llm_response: str = "") -> str:
    if action_id == 0:  # ALLOW
        return llm_response
    elif action_id == 1:  # BLOCK
        return "Your request has been blocked due to policy violation."
    elif action_id == 2:  # REWRITE
        return rewrite_prompt_and_generate(prompt)
    elif action_id == 3:  # ASK
        return "I'm not sure how to proceed. Can you clarify your intent?"
    elif action_id == 4:  # ESCALATE
        time.sleep(random.uniform(0.5, 1.5))
        return "[Simulated human moderator review]"

OBS_LENGTH = 393  # embedding input size for ONNX

for ds_info in datasets_info:
    print(f"\n🔎 Processing dataset: {ds_info['name']}")
    dataset = load_dataset(ds_info["name"], split=ds_info["split"])
    actions = []
    results = []
    # Always sample a random 1000 examples (or all if <1000)
    num_samples = min(len(dataset), 1000)
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in tqdm(indices, desc=f"Inference [{ds_info['name']}]"):
        example = dataset[idx]
        prompt = example.get(ds_info["prompt_key"], "")
        completion = example.get("completion", "")

        # Encode prompt
        prompt_embedding = embedder.encode(prompt).astype(float).tolist()
        if len(prompt_embedding) < OBS_LENGTH:
            prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
        obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

        # ONNX model predicts action
        outputs = session.run(None, {session.get_inputs()[0].name: obs})
        action_id = int(np.argmax(outputs[0], axis=1)[0])

        # Execute action
        start_time = time.time()
        final_response = execute_action(prompt, action_id, completion)
        latency = (time.time() - start_time) * 1000

        result = {
            "index": idx,
            "dataset": ds_info["name"],
            "prompt": prompt,
            "label": 0,
            "action_taken": action_names[action_id],
            "action_id": action_id,
            "final_response": final_response,
            "latency_ms": latency
        }
        results.append(result)
        actions.append(action_id)
        all_results.append(result)
        all_actions.append(action_id)
    # Save per-dataset results
    outname = f"DEMODISTCOPY.json"
    with open(outname, "w") as f:
        json.dump(results, f, indent=2)
    per_dataset_actions.append(actions)
    per_dataset_names.append(ds_info["name"])

from collections import Counter
# Print per-dataset action distributions and FPR
for ds_name, actions in zip(per_dataset_names, per_dataset_actions):
    action_counts = Counter(actions)
    print(f"\n📊 Action Distribution for {ds_name}:")
    for i, count in sorted(action_counts.items()):
        print(f"  {action_names[i]} ({i}): {count}")
    # False positive rate: BLOCK / total
    total = sum(action_counts.values())
    block_count = action_counts.get(1, 0)
    fpr = block_count / total if total > 0 else 0
    print(f"  False Positive Rate (BLOCK): {fpr:.4f}")

# Print combined action distribution and FPR
action_counts = Counter(all_actions)
print("\n📊 Combined Action Distribution:")
for i, count in sorted(action_counts.items()):
    print(f"{action_names[i]} ({i}): {count}")
total = sum(action_counts.values())
block_count = action_counts.get(1, 0)
fpr = block_count / total if total > 0 else 0
print(f"False Positive Rate (BLOCK): {fpr:.4f}")
