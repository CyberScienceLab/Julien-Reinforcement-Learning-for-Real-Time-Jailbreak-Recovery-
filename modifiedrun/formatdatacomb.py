from datasets import load_dataset
from datasets import Dataset
import ollama
import re
from tqdm import tqdm

behaviors = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
judge_cmp = load_dataset("JailbreakBench/JBB-Behaviors", "judge_comparison")
#toxicity_ds = load_dataset("allenai/real-toxicity-prompts")
ands = load_dataset("Anthropic/hh-rlhf", split="train")
advbench = load_dataset("walledai/AdvBench")
awesome_chatgpt = load_dataset("fka/awesome-chatgpt-prompts")
def reformat_awesome_chatgpt(ds):
    examples = []
    for ex in ds['train']:
        prompt = ex.get('prompt', '')
        examples.append({
            'prompt': prompt,
            'label': 0,
            'source': 'awesome_chatgpt_prompts',
            'goal': prompt,
            'target': '',
            'metadata': {}
        })
    return examples

def reformat_behaviors(behaviors_ds):
    examples = []
    for split in ['harmful', 'benign']:
        label = 1 if split == 'harmful' else 0
        for ex in behaviors_ds[split]:
            prompt = ex.get('Goal', '')
            examples.append({
                'prompt': prompt,
                'label': label,
                'source': 'behaviors',
                'goal': prompt,
                'target': ex.get('Target', ''),
                'metadata': {
                    'Behavior': ex.get('Behavior'),
                    'Category': ex.get('Category'),
                    'Index': ex.get('Index'),
                    'Source': ex.get('Source')
                }
            })
    return examples

def reformat_advbench(advbench_ds):
    examples = []
    #print(advbench_ds)
    for ex in advbench_ds['train']:
        #label = int(ex.get('human_majority', 0))
        prompt = ex.get('prompt', '')
        examples.append({
            'prompt': prompt,
            'label': 1,
            'source': 'advbench',
            'goal': prompt,
            'target': ex.get('target_response', ''),
            'metadata': {
                'dataset': "advbench"
                
            }
        })
    return examples


def reformat_judge_comparison(judge_ds):
    examples = []
    for ex in judge_ds['test']:
        label = int(ex.get('human_majority', 0))
        prompt = ex.get('prompt', '')
        examples.append({
            'prompt': prompt,
            'label': label,
            'source': 'judge_comparison',
            'goal': prompt,
            'target': ex.get('target_response', ''),
            'metadata': {
                'Index': ex.get('Index'),
                'human1': ex.get('human1'),
                'human2': ex.get('human2'),
                'human3': ex.get('human3'),
                'harmbench_cf': ex.get('harmbench_cf'),
                'gpt4_cf': ex.get('gpt4_cf'),
                'llamaguard2_cf': ex.get('llamaguard2_cf'),
                'llama3_cf': ex.get('llama3_cf'),
            }
        })
    return examples

def extract_all_human_prompts(text):
    return [p.strip() for p in re.findall(r'Human:\s*(.*?)(?=\n*\s*Assistant:|\n*\s*Human:|$)', text, re.DOTALL)]
'''
def ollama_toxicity_classify(prompt: str, model_name='mistral'):
    instruction = f"Classify the toxicity of this text as either 'safe' or 'unsafe'. Respond with exactly one word:\n\n{prompt}"
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": instruction}])
    return response['message']['content'].strip().lower()
'''
def ollama_toxicity_classify(prompt: str, model_name='mistral'):
    instruction = (
        "Classify the toxicity of this text as either 'safe', 'unsafe', or 'unsure'. "
        "Respond with exactly one word:\n\n" + prompt
    )
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": instruction}])
    return response['message']['content'].strip().lower()

def reformat_anthropic_hhrlhf(dataset):
    examples = []
    #get only the first 1000
    #dataset = dataset.select(range(1000))  # Limit to first 1000 examples for performance
    dataset = dataset.shuffle(seed=42).select(range(250))
    for ex in tqdm(dataset, desc="Processing Anthropic HH-RLHF dataset"):
        chosen_text = ex['chosen']
        rejected_text = ex['rejected']
        human_prompts = extract_all_human_prompts(chosen_text)
        for prompt in human_prompts:
            try:
                label_str = ollama_toxicity_classify(prompt)
                if "unsafe" in label_str:
                    label = 1
                elif "safe" in label_str:
                    label = 0
                else:
                    label = -1
            except Exception as e:
                continue
            examples.append({
                'prompt': prompt,
                'label': label,
                'source': 'anthropic_hhrlhf',
                'goal': prompt,
                'target': ex.get('chosen', ''),
                'metadata': {"chosen": chosen_text, "rejected": rejected_text}
            })
    return examples

behaviors_reformatted = reformat_behaviors(behaviors)
judge_reformatted = reformat_judge_comparison(judge_cmp)
anthropic_reformatted = reformat_anthropic_hhrlhf(ands)
advbench_re = reformat_advbench(advbench)
awesome_chatgpt_re = reformat_awesome_chatgpt(awesome_chatgpt)

combined_examples = behaviors_reformatted + judge_reformatted + anthropic_reformatted + advbench_re + awesome_chatgpt_re

# Step 3: Create a Hugging Face Dataset from the combined list
combined_dataset = Dataset.from_list(combined_examples)

# Step 4 (optional): shuffle for randomness
combined_dataset = combined_dataset.shuffle(seed=42)

# Step 5: Inspect some examples
print(f"Total examples: {len(combined_dataset)}")
print("\nFirst example:")
print(combined_dataset[0])
print("\nLast example:")
print(combined_dataset[-1])

# Step 6 (optional): save locally for later use
combined_dataset.save_to_disk('./FINALDATA')

print(f"\nDataset saved with {len(behaviors_reformatted)} behaviors, {len(judge_reformatted)} judge comparisons, {len(advbench_re)} advbench responses, {len(anthropic_reformatted)} anthropic prompts, and {len(awesome_chatgpt_re)} awesome_chatgpt prompts")