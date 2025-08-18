import json
import numpy as np
from collections import Counter
from stable_baselines3 import PPO
from datasets import load_from_disk
from envtest import JailbreakDefenseEnv
import ollama
import time
import random
from tqdm import tqdm

# Load dataset and model
dataset = load_from_disk('COMPLETE_DATAMOD')
model = PPO.load("DEMO")

action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
results = []
actions = []


def rewrite_prompt_and_generate(prompt: str, model='mistral') -> str:
    instruction = f"Rewrite the following prompt to make it safer and more appropriate, while preserving the core meaning. You may summarize or rephrase as needed:\n\n{prompt}"
    response = ollama.chat(model=model, messages=[{"role": "user", "content": instruction}])
    return response['message']['content']

def execute_action(prompt: str, llm_response: str, action_id: int) -> str:
    if action_id == 0:  # ALLOW
        return llm_response
    elif action_id == 1:  # BLOCK
        return "Your request has been blocked due to policy violation."
    elif action_id == 2:  # REWRITE
        return rewrite_prompt_and_generate(prompt)
    elif action_id == 3:  # ASK
        return "I'm not sure how to proceed. Can you clarify your intent?"
    elif action_id == 4:  # ESCALATE
        #print("\n *** HUMAN MODERATOR REVIEW REQUIRED ***")
        #print(f"Prompt: {prompt}")
        #print(f"Model Response: {llm_response}")
        human_input = "Simulate human input"#input("Enter your response or action here: ")
        time.sleep(random.uniform(0.5, 4.5))  # Simulate some processing time
        return human_input

num_examples = len(dataset)#int(0.1 * len(dataset))
for i in tqdm(range(num_examples), desc="Running Inference"):
    example = [dataset[i]]
    # Check for valid example type and non-empty
    if not example or not isinstance(example[0], dict):
        print(f"Skipping index {i}: invalid or empty example.")
        continue
    difficulties = ["easy", "medium", "hard"]
    env = None
    obs = None
    for diff in difficulties:
        try:
            env = JailbreakDefenseEnv(example, difficulty=diff)
            obs, _ = env.reset()
            break
        except Exception as e:
            continue
    if env is None or obs is None:
        print(f"Skipping index {i}: no valid difficulty for example.")
        continue

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)

    start_time = time.time()
    label = example[0]['label'] if 'label' in example[0] else example[0].get('target_contains_policy_violation', 0)
    promptguard = execute_action(example[0]["prompt"], example[0]["target"], int(action))
    latency = (time.time() - start_time) * 1000

    result = {
        "index": i,
        "prompt": example[0]["prompt"],
        "label": label,
        "action_taken": action_names[action],
        "action_id": int(action),
        "reward": float(reward),
        "ground_truth_violation": bool(example[0]['target_contains_policy_violation']),
        "final_response": promptguard,
        "latency_ms": latency,
        "step_info": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()}
    }
    results.append(result)
    actions.append(int(action))

tp = fp = tn = fn = 0
for res in results:
    predicted_violation = res["action_id"] in [1, 2, 3, 4]  # non-ALLOW
    ground_truth_violation = res["label"] == 1 or res["ground_truth_violation"]

    if ground_truth_violation:
        if predicted_violation:
            tp += 1  # true positive
        else:
            fn += 1  # false negative
    else:
        if predicted_violation:
            fp += 1  # false positive
        else:
            tn += 1  # true negative

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\n📈 TPR / FPR Metrics:")
print(f"True Positives: {tp}")
print(f"False Positives: {fp}")
print(f"True Negatives: {tn}")
print(f"False Negatives: {fn}")
print(f"TPR (Recall): {tpr:.3f}")
print(f"FPR: {fpr:.3f}")

# Save to JSON file
with open("inference_resultsADV.json", "w") as f:
    json.dump(results, f, indent=2)

# Print action distribution
action_counts = Counter(actions)
print("\n📊 Action Distribution:")
for i, count in sorted(action_counts.items()):
    print(f"{action_names[i]} ({i}): {count}")
