import re
import random
from datasets import load_from_disk, Dataset
from better_profanity import profanity
from scipy.stats import entropy
from collections import Counter
import numpy as np

DATA_PATH = './COMPLETE_DATA'
OUTPUT_PATH = './COMPLETE_DATA_ANTHROPIC_ASSISTANT_ONLY'

# Load the dataset
print(f"Loading dataset from {DATA_PATH}...")
dataset = load_from_disk(DATA_PATH)

# Helper functions for feature extraction
REFUSAL_PHRASES = [
    "I can't help with that.", "I'm not able to assist.", "I can't provide that.", "I cannot help with this.",
    "I'm unable to do that.", "That’s not something I can do.", "I won’t be able to assist.", "I must decline.",
    "I’m sorry, no.", "That’s outside my scope."
]
def contains_refusal(text):
    return int(any(phrase in text.lower() for phrase in REFUSAL_PHRASES))
def response_length(text):
    return len(text.split())
def char_entropy(text):
    if not text:
        return 0.0
    counts = Counter(text)
    probs = [count / len(text) for count in counts.values()]
    return float(entropy(probs, base=2))
def contains_profanity(text):
    return int(profanity.contains_profanity(text))

def extract_assistant_response(prompt, target):
    # Find last occurrence of prompt
    prompt_idx = target.lower().rfind(prompt.lower())
    if prompt_idx == -1:
        return ""  # fallback: no match
    after_prompt = target[prompt_idx + len(prompt):]
    # Find the next 'Assistant:' after the prompt
    match = re.search(r'Assistant:(.*?)(?:\nHuman:|$)', after_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""  # fallback: no assistant response found

# Filter for Anthropic examples (assuming a 'source' field exists)
def is_anthropic(example):
    return example.get('source', '').lower() == 'anthropic_hhrlhf'

# Process and update Anthropic examples
def process_anthropic_examples(dataset):
    new_examples = []
    for ex in dataset:
        if not is_anthropic(ex):
            continue
        prompt = ex.get('prompt', '')
        target = ex.get('target', '')
        assistant_response = extract_assistant_response(prompt, target)
        # Recompute target-specific features
        refusal = contains_refusal(assistant_response)
        resp_len = response_length(assistant_response)
        entropy_val = char_entropy(assistant_response)
        profanity_val = contains_profanity(assistant_response)
        # Copy and update
        new_ex = dict(ex)
        new_ex['target'] = assistant_response
        new_ex['target_contains_refusal'] = refusal
        new_ex['target_response_length'] = resp_len
        new_ex['target_char_entropy'] = entropy_val
        new_ex['prompt_contains_profanity'] = profanity_val
        new_examples.append(new_ex)
    return new_examples

print("Processing Anthropic examples...")
processed = process_anthropic_examples(dataset)
print(f"Processed {len(processed)} Anthropic examples.")

# Save as new dataset
if processed:
    new_dataset = Dataset.from_list(processed)
    new_dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved updated Anthropic assistant-only dataset to {OUTPUT_PATH}")
else:
    print("No Anthropic examples found to process.")
