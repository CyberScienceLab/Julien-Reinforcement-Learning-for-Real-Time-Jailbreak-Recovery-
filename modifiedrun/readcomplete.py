import random
from datasets import load_from_disk

# Path to your processed dataset
DATA_PATH = './COMPLETE_DATAMOD'

# Load the dataset
print(f"Loading dataset from {DATA_PATH}...")
dataset = load_from_disk(DATA_PATH)

# Filter for Anthropic examples (assuming a 'source' field exists)
def is_anthropic(example):
    return example.get('source', '').lower() == 'anthropic_hhrlhf'

# Get all Anthropic examples
anthropic_examples = [ex for ex in dataset if is_anthropic(ex)]

print(f"Found {len(anthropic_examples)} Anthropic examples.")

# Show a few random samples
num_to_show = min(5, len(anthropic_examples))
print(f"Showing {num_to_show} random Anthropic examples:")
for ex in random.sample(anthropic_examples, num_to_show):
    print("---")
    print(f"Prompt: {ex.get('prompt')}")
    print(f"Target: {ex.get('target')}")
    print(f"Difficulty: {ex.get('difficulty')}")
    print(f"Policy Violation: {ex.get('target_contains_policy_violation')}")
    print(f"Jailbreak Likelihood: {ex.get('prompt_jailbreak_likelihood')}")
    print(f"System Role Token Hit: {ex.get('prompt_system_role_token_hit')}")
    print(f"Refusal: {ex.get('target_contains_refusal')}")
    print(f"Response Length: {ex.get('target_response_length')}")
    print(f"Char Entropy: {ex.get('target_char_entropy')}")
    print(f"Profanity: {ex.get('prompt_contains_profanity')}")
    print(f"Policy Violation Logit: {ex.get('policy_violation_logit')}")
    print(f"Guardrail Flagged: {ex.get('wrapper_flagged')}")
    print(f"Simulated Latency (ms): {ex.get('simulated_latency_ms')}")
    print(f"Simulated Escalate Output: {ex.get('simulated_escalate_output')}")
    print(f"Simulated Ask Output: {ex.get('simulated_ask_output')}")
    print(f"Simulated Rewrite Output: {ex.get('simulated_rewrite_output')}")
    print(f"Rewrite Preserves Intent: {ex.get('rewrite_preserves_intent')}")
    print("---\n")
