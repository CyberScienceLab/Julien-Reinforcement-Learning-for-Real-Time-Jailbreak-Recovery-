from datasets import load_dataset
from detoxify import Detoxify
import re
import ollama

# Load Anthropic HH-RLHF dataset
dataset = load_dataset("Anthropic/hh-rlhf", split="train")

detox = Detoxify('original')

def extract_all_human_prompts(text):
    # Find all segments starting with "Human:" and ending before next "Assistant:" or "Human:"
    prompts = re.findall(r'Human:\s*(.*?)(?=\n\nAssistant:|\n\nHuman:|$)', text, re.DOTALL)
    # Clean whitespace from each prompt
    return [p.strip() for p in prompts]


def ollama_toxicity_classify(prompt: str, model_name='mistral'):
    instruction = f"Classify the toxicity of this text as either 'safe' or 'unsafe'. Respond with exactly one word:\n\n{prompt}"
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": instruction}])
    return response['message']['content'].strip().lower()


print("Printing first 5 human prompts with Detoxify scores and Ollama classification:\n")

for i, example in enumerate(dataset.select(range(20))):
    chosen_text = example['chosen']
    human_prompts = extract_all_human_prompts(chosen_text)
    if not human_prompts:
        continue

    print(f"Example {i + 1}:")
    for j, human_prompt in enumerate(human_prompts):
        detox_scores = detox.predict(human_prompt)
        max_toxicity = max(detox_scores.values())
        try:
            ollama_result = ollama_toxicity_classify(human_prompt)
        except Exception as e:
            ollama_result = f"Error: {e}"

        print(f"  Human Prompt {j + 1}: {human_prompt}")
        print(f"  Detoxify Toxicity Score: {max_toxicity:.3f}")
        print(f"  Ollama Toxicity Classification: {ollama_result}")
        print()
    print("-" * 40)
