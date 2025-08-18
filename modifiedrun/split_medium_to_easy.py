import numpy as np
from datasets import load_from_disk, Dataset
from collections import Counter

# Load processed dataset
input_path = 'FINALCOMPLETE_DATA'
dataset = load_from_disk(input_path)

# Get all medium examples and their difficulty scores
medium_indices = [i for i, d in enumerate(dataset['difficulty']) if d == 'medium']
medium_scores = []
for i in medium_indices:
    jb_score = dataset[i]['prompt_jailbreak_likelihood'] if 'prompt_jailbreak_likelihood' in dataset[i] else 0
    ent = dataset[i]['target_char_entropy'] if 'target_char_entropy' in dataset[i] else 0
    score = (jb_score * 0.6) + (ent / 5.0 * 0.4)
    medium_scores.append((i, score))

# Sort medium examples by difficulty score
medium_scores.sort(key=lambda x: x[1])

# Select 900 easiest as 'easy', rest as 'medium'
easy_indices = [idx for idx, _ in medium_scores[:900]]
new_medium_indices = [idx for idx, _ in medium_scores[900:]]

# Update difficulty labels
new_difficulty = list(dataset['difficulty'])
for idx in easy_indices:
    new_difficulty[idx] = 'easy'
for idx in new_medium_indices:
    new_difficulty[idx] = 'medium'

dataset = dataset.remove_columns('difficulty')
dataset = dataset.add_column('difficulty', new_difficulty)

# Save to new file
output_path = 'FINALCOMPLETE_DATA_SPLIT'
dataset.save_to_disk(output_path)
print(f"Saved new dataset to {output_path}")
print("Difficulty distribution:", Counter(new_difficulty))
