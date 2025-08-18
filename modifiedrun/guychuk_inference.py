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

DATASET_NAME = "rubend18/ChatGPT-Jailbreak-Prompts"
PROMPT_KEY = "prompt"
SPLIT = "train"

session = ort.InferenceSession("DEMO.onnx")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
results = []
actions = []
labels = []

OBS_LENGTH = 393  # embedding input size for ONNX

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

print(f"\n🔎 Processing dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME, split=SPLIT)

def run_inference_on_dataset(dataset, prompt_key, label_key=None, malicious_only=False, max_samples=1000):
    results = []
    actions = []
    labels = []
    num_samples = min(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    for idx in tqdm(indices, desc=f"Inference"):
        example = dataset[idx]
        prompt = example.get(prompt_key, "")
        completion = example.get("completion", "")
        label = int(example.get(label_key, 1)) if label_key else 1

        prompt_embedding = embedder.encode(prompt).astype(float).tolist()
        if len(prompt_embedding) < OBS_LENGTH:
            prompt_embedding += [0.0] * (OBS_LENGTH - len(prompt_embedding))
        obs = np.array(prompt_embedding, dtype=np.float32).reshape(1, -1)

        outputs = session.run(None, {session.get_inputs()[0].name: obs})
        action_id = int(np.argmax(outputs[0], axis=1)[0])

        start_time = time.time()
        final_response = execute_action(prompt, action_id, completion)
        latency = (time.time() - start_time) * 1000

        result = {
            "index": idx,
            "prompt": prompt,
            "label": label,
            "action_taken": action_names[action_id],
            "action_id": action_id,
            "final_response": final_response,
            "latency_ms": latency
        }
        results.append(result)
        actions.append(action_id)
        labels.append(label)
    return results, actions, labels

# Run on rubend18/ChatGPT-Jailbreak-Prompts (malicious only)
print(f"\n🔎 Processing dataset: rubend18/ChatGPT-Jailbreak-Prompts")
dataset_jailbreak = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
results_jailbreak, actions_jailbreak, _ = run_inference_on_dataset(dataset_jailbreak, "prompt", malicious_only=True)

# Run on xTRam1/safe-guard-prompt-injection (safe/malicious)
print(f"\n🔎 Processing dataset: xTRam1/safe-guard-prompt-injection")
dataset_guard = load_dataset("xTRam1/safe-guard-prompt-injection", split="train")
results_guard, actions_guard, labels_guard = run_inference_on_dataset(dataset_guard, "text", "label", malicious_only=False)


# Save results
with open("GUYCHUKINF_onnx_jailbreak.json", "w") as f:
    json.dump(results_jailbreak, f, indent=2)
with open("GUYCHUKINF_onnx_guard.json", "w") as f:
    json.dump(results_guard, f, indent=2)

# Action distribution and metrics for jailbreak dataset
action_counts_jailbreak = Counter(actions_jailbreak)
print("\n📊 Action Distribution (Jailbreak):")
for i, count in sorted(action_counts_jailbreak.items()):
    print(f"{action_names[i]} ({i}): {count}")
num_non_allow_jailbreak = sum(1 for a in actions_jailbreak if a != 0)
total_jailbreak = len(actions_jailbreak)
print(f"\nTPR (non-ALLOW rate, Jailbreak): {num_non_allow_jailbreak / total_jailbreak if total_jailbreak > 0 else 0:.4f}")

# Action distribution and metrics for safe-guard dataset
action_counts_guard = Counter(actions_guard)
print("\n📊 Action Distribution (Safe-Guard):")
for i, count in sorted(action_counts_guard.items()):
    print(f"{action_names[i]} ({i}): {count}")

# TPR, FPR, etc. for safe-guard dataset
TP = sum(1 for a, l in zip(actions_guard, labels_guard) if a != 0 and l == 1)  # non-ALLOW & malicious
FP = sum(1 for a, l in zip(actions_guard, labels_guard) if a != 0 and l == 0)  # non-ALLOW & safe
TN = sum(1 for a, l in zip(actions_guard, labels_guard) if a == 0 and l == 0)  # ALLOW & safe
FN = sum(1 for a, l in zip(actions_guard, labels_guard) if a == 0 and l == 1)  # ALLOW & malicious
total_guard = len(labels_guard)
print(f"\nConfusion Matrix Stats (Safe-Guard):")
print(f"TP (non-ALLOW & malicious): {TP}")
print(f"FP (non-ALLOW & safe): {FP}")
print(f"TN (ALLOW & safe): {TN}")
print(f"FN (ALLOW & malicious): {FN}")
print(f"TPR (Recall): {TP / (TP + FN) if (TP + FN) > 0 else 0:.4f}")
print(f"FPR: {FP / (FP + TN) if (FP + TN) > 0 else 0:.4f}")
print(f"TNR (Specificity): {TN / (TN + FP) if (TN + FP) > 0 else 0:.4f}")
print(f"FNR: {FN / (FN + TP) if (FN + TP) > 0 else 0:.4f}")
print(f"Accuracy: {(TP + TN) / total_guard if total_guard > 0 else 0:.4f}")

# Combined analysis
actions_combined = actions_jailbreak + actions_guard
labels_combined = [1]*len(actions_jailbreak) + labels_guard
TP_c = sum(1 for a, l in zip(actions_combined, labels_combined) if a != 0 and l == 1)
FP_c = sum(1 for a, l in zip(actions_combined, labels_combined) if a != 0 and l == 0)
TN_c = sum(1 for a, l in zip(actions_combined, labels_combined) if a == 0 and l == 0)
FN_c = sum(1 for a, l in zip(actions_combined, labels_combined) if a == 0 and l == 1)
total_c = len(labels_combined)
print(f"\nConfusion Matrix Stats (Combined):")
print(f"TP (non-ALLOW & malicious): {TP_c}")
print(f"FP (non-ALLOW & safe): {FP_c}")
print(f"TN (ALLOW & safe): {TN_c}")
print(f"FN (ALLOW & malicious): {FN_c}")
print(f"TPR (Recall): {TP_c / (TP_c + FN_c) if (TP_c + FN_c) > 0 else 0:.4f}")
print(f"FPR: {FP_c / (FP_c + TN_c) if (FP_c + TN_c) > 0 else 0:.4f}")
print(f"TNR (Specificity): {TN_c / (TN_c + FP_c) if (TN_c + FP_c) > 0 else 0:.4f}")
print(f"FNR: {FN_c / (FN_c + TP_c) if (FN_c + TP_c) > 0 else 0:.4f}")
print(f"Accuracy: {(TP_c + TN_c) / total_c if total_c > 0 else 0:.4f}")
