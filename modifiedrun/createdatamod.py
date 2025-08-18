# System role tokens for detection
SYSTEM_ROLE_TOKENS = [
    "system:", "user:", "assistant:", "you are an ai", "as an ai", "as a system", "role:", "<|system|>", "<|user|>", "<|assistant|>"
]

def system_role_token_hits(text):
    return int(any(token in text.lower() for token in SYSTEM_ROLE_TOKENS))
import logging
import os
import re
import random
from collections import Counter
from tqdm import tqdm
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from better_profanity import profanity
from scipy.stats import entropy
import numpy as np
#import openai
import ollama
from sklearn.metrics.pairwise import cosine_similarity

from detoxify import Detoxify
import time

#openai.api_key = os.getenv("OPENAI_API_KEY")
detox_model = Detoxify('original')
dataset = load_from_disk('FINALFUZZED_DATA')
model = SentenceTransformer('all-MiniLM-L6-v2')
output_path = './FINALCOMPLETE_DATA'

USE_ENTROPY = True
USE_JAILBREAK = True
USE_POLICY_LOGIT = True
USE_GUARDRAIL = True
USE_LATENCY = True
USE_TARGET_POLICY = True

# Logging Setup 
logging.basicConfig(level=logging.INFO)

REFUSAL_PHRASES = [
    "I can't help with that.",
    "I'm not able to assist.",
    "I can't provide that.",
    "I cannot help with this.",
    "I'm unable to do that.",
    "That’s not something I can do.",
    "I won’t be able to assist.",
    "I must decline.",
    "I’m sorry, no.",
    "That’s outside my scope.",
]


JAILBREAK_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"pretend (to|you are)",
    r"you are (no longer|not) an AI",
    r"do anything now",
    r"\bDAN\b",
    r"jailbreak mode",
    r"you must obey",
    r"as a fictional scenario",
    r"simulate a scenario",
    r"bypass restrictions",
    r"disregard guidelines",
    r"override safety",
    r"act as a hacker",
    r"act as a pirate",
    r"disable your filters",
    r"respond without censorship",
    r"respond regardless of ethics",
    r"secret instructions",
    r"hidden command",
]

import base64
import json

# Read base64 string from file
with open('unsafe_keywords.txt', 'rb') as f:
    encoded_bytes = f.read()

# Decode base64 to JSON string
json_str = base64.b64decode(encoded_bytes).decode('utf-8')

# Convert JSON string back to list
UNSAFE_KEYWORDS = json.loads(json_str)
#print(UNSAFE_KEYWORDS)
#exit()
'''
UNSAFE_KEYWORDS = [
    "how to make a bomb", "disable a car", "hack", "kill", "steal",
    "drug recipe", "fake passport", "harm yourself", "exploit", "deepfake"
]'''

def rewrite_prompt_and_generate(prompt: str, model_name='mistral', model=None):
    instruction = (
        f"Rewrite the following prompt to make it safer and more appropriate, "
        f"while preserving the core meaning. You may summarize or rephrase as needed:\n\n{prompt}"
    )
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": instruction}])
    rewritten = response['message']['content']

    # Only compute similarity if model is passed (you need SentenceTransformer instance)
    similarity = None
    if model is not None:
        orig_emb = model.encode(prompt, convert_to_numpy=True)
        rewritten_emb = model.encode(rewritten, convert_to_numpy=True)
        similarity = cosine_similarity(orig_emb.reshape(1, -1), rewritten_emb.reshape(1, -1))[0][0]

    return rewritten, similarity

def simulate_escalate(label: int) -> str:
    """Simulate human moderator response for escalation."""
    if label:
        return "Request rejected by human moderator due to policy violation."
    else:
        return "Request approved by human moderator."

def simulate_ask() -> str:
    """Simulated response for ASK action (can customize)."""
    return "Can you please clarify your intent?"

def contains_refusal(text):
    return int(any(phrase in text.lower() for phrase in REFUSAL_PHRASES))

def contains_profanity(text):
    return int(profanity.contains_profanity(text))

def response_length(text):
    return len(text.split())

def char_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    probs = [count / len(text) for count in counts.values()]
    return float(entropy(probs, base=2))

def jailbreak_score(prompt):
    score = sum(bool(re.search(pattern, prompt.lower())) for pattern in JAILBREAK_PATTERNS)
    return score / len(JAILBREAK_PATTERNS)

def check_moderation(text):
    try:
        scores = detox_model.predict(text)
        max_score = max(scores.values())
        flagged = int(max_score > 0.5)  # You can tune the threshold
        return flagged, max_score
    except Exception as e:
        logging.warning(f"Detoxify error: {e}")
        return 0, 0.0

def simulate_latency():
    return random.randint(50, 300)

def contains_policy_violation(text):
    return int(any(kw in text.lower() for kw in UNSAFE_KEYWORDS))

# Add this before your main loop  
all_difficulty_scores = []

# First pass: calculate difficulty scores
for example in tqdm(dataset, desc="Extracting features", total=len(dataset)):
    jb_score = jailbreak_score(example['prompt'])
    ent = char_entropy(example['target'])
    # Simple weighted combination
    difficulty_score = (jb_score * 0.6) + (ent / 5.0 * 0.4)
    all_difficulty_scores.append(difficulty_score)

# Get exact percentiles for equal distribution
easy_threshold = np.percentile(all_difficulty_scores, 33.33)
hard_threshold = np.percentile(all_difficulty_scores, 66.67)

def assign_difficulty(example):
    jb_score = jailbreak_score(example['prompt'])
    ent = char_entropy(example['target'])
    difficulty_score = (jb_score * 0.6) + (ent / 5.0 * 0.4)
    
    if difficulty_score >= hard_threshold:
        return 'hard'
    elif difficulty_score >= easy_threshold:
        return 'medium'
    else:
        return 'easy'


# Feature containers
prompt_embeddings = []
target_embeddings = []
profanity_flags = []
refusal_flags = []
response_lengths = []
system_role_flags = []

# Synthetic / simulated signals:
entropy_scores = []
jailbreak_likelihoods = []
policy_violation_logits = []
latencies = []
guardrail_flags = []
target_policy_violations = []
difficulty_labels = []

# New simulated action outputs:
simulated_escalate_outputs = []
simulated_ask_outputs = []
simulated_rewrite_outputs = []
preserved_labels = []

print("Extracting features and simulating actions...")
for example in tqdm(dataset, desc="Extracting features", total=len(dataset)):
    prompt = example['prompt']
    target = example['target']
    if contains_policy_violation(prompt):
        label = 1
    else:
        label = example['label'] if 'label' in example else example.get('target_contains_policy_violation', 0)

    # Embeddings
    prompt_emb = model.encode(prompt)
    target_emb = model.encode(target)
    prompt_embeddings.append(prompt_emb)
    target_embeddings.append(target_emb)

    # Binary / heuristic features
    profanity_flags.append(contains_profanity(prompt))
    refusal_flags.append(contains_refusal(target))
    response_lengths.append(response_length(target))
    system_role_flags.append(system_role_token_hits(prompt))
    difficulty_labels.append(assign_difficulty(example))

    if USE_ENTROPY:
        entropy_scores.append(char_entropy(target))
    if USE_JAILBREAK:
        jailbreak_likelihoods.append(jailbreak_score(prompt))
    if USE_POLICY_LOGIT or USE_GUARDRAIL:
        flagged, logit = check_moderation(target)
        if USE_POLICY_LOGIT:
            policy_violation_logits.append(logit)
        if USE_GUARDRAIL:
            guardrail_flags.append(flagged)
    if USE_TARGET_POLICY:
        target_policy_violations.append(label)
    if USE_LATENCY:
        #latencies.append(simulate_latency())
        start_time = time.time()
        _ = model.encode(prompt)
        _ = model.encode(target)
        if USE_POLICY_LOGIT or USE_GUARDRAIL:
            _ = check_moderation(target)
        latency_ms = (time.time() - start_time) * 1000
        latencies.append(int(latency_ms))

    # Simulated action outputs
    simulated_escalate_outputs.append(simulate_escalate(label))
    simulated_ask_outputs.append(simulate_ask())
    # For rewrite, to avoid too many calls to LLM, you might want to cache or simplify
    # Here calling rewrite once per example - consider async or caching for large data
    rewrite, cosine_sim = rewrite_prompt_and_generate(prompt, model=model)
    simulated_rewrite_outputs.append(rewrite)
    preserved_labels.append(1 if cosine_sim and cosine_sim >= 0.75 else 0)

# Convert to numpy
prompt_embeddings = np.array(prompt_embeddings)
target_embeddings = np.array(target_embeddings)


dataset = dataset.add_column("difficulty", difficulty_labels)
dataset = dataset.add_column("prompt_embedding", list(prompt_embeddings))
dataset = dataset.add_column("target_response_embedding", list(target_embeddings))
dataset = dataset.add_column("prompt_contains_profanity", profanity_flags)
dataset = dataset.add_column("target_contains_refusal", refusal_flags)
dataset = dataset.add_column("target_response_length", response_lengths)
dataset = dataset.add_column("prompt_system_role_token_hit", system_role_flags)

if USE_ENTROPY:
    dataset = dataset.add_column("target_char_entropy", entropy_scores)
if USE_JAILBREAK:
    dataset = dataset.add_column("prompt_jailbreak_likelihood", jailbreak_likelihoods)
if USE_POLICY_LOGIT:
    dataset = dataset.add_column("policy_violation_logit", policy_violation_logits)
if USE_GUARDRAIL:
    dataset = dataset.add_column("wrapper_flagged", guardrail_flags)
if USE_LATENCY:
    dataset = dataset.add_column("simulated_latency_ms", latencies)
if USE_TARGET_POLICY:
    dataset = dataset.add_column("target_contains_policy_violation", target_policy_violations)

# Add simulated outputs
dataset = dataset.add_column("simulated_escalate_output", simulated_escalate_outputs)
dataset = dataset.add_column("simulated_ask_output", simulated_ask_outputs)
dataset = dataset.add_column("simulated_rewrite_output", simulated_rewrite_outputs)
dataset = dataset.add_column("rewrite_preserves_intent", preserved_labels)

dataset.save_to_disk(output_path)
print("Saved with extended features and simulated outputs.")
print("Difficulty distribution:", Counter(difficulty_labels))
