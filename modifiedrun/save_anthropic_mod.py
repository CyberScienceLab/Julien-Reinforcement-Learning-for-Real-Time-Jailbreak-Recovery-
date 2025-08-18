import re
from datasets import load_from_disk, Dataset, DatasetDict
from better_profanity import profanity
from scipy.stats import entropy
from collections import Counter
import numpy as np
from tqdm import tqdm

DATA_PATH = './COMPLETE_DATA'
OUTPUT_PATH = './COMPLETE_DATAMOD'

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
    for ex in tqdm(dataset, desc="Processing examples"):
        if is_anthropic(ex):
            prompt = ex.get('prompt', '')
            target = ex.get('target', '')
            assistant_response = extract_assistant_response(prompt, target)
            # Only update target and target-specific features
            new_ex = dict(ex)
            new_ex['target'] = assistant_response
            new_ex['target_contains_refusal'] = contains_refusal(assistant_response)
            new_ex['target_response_length'] = response_length(assistant_response)
            new_ex['target_char_entropy'] = char_entropy(assistant_response)
            # All other columns remain unchanged
            new_examples.append(new_ex)
        else:
            new_examples.append(dict(ex))
    return new_examples

print("Processing dataset and updating Anthropic examples only...")
processed = process_anthropic_examples(dataset)
print(f"Processed {len(processed)} total examples.")

# Save as new dataset
if processed:
    new_dataset = Dataset.from_list(processed)
    new_dataset.save_to_disk(OUTPUT_PATH)
    print(f"Saved updated dataset to {OUTPUT_PATH}")
else:
    print("No examples found to process.")
