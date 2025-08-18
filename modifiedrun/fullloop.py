from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from datasets import load_from_disk
from envtest import JailbreakDefenseEnv  # Your custom environment
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env(dataset, difficulty):
    def _init():
        return JailbreakDefenseEnv(dataset, difficulty=difficulty)
    return _init

# --- Evaluation Callback ---
class EvaluationCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5000, patience=3, delta=0.005, verbose=1, eval_episodes=500):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.delta = delta
        self.recent_rewards = []
        self.recent_tprs = []
        self.eval_episodes = eval_episodes

        # Pre-sample a fixed list of indices for consistent evaluation
        self.eval_indices = []
        
        # Unwrap the environment
        env = self.eval_env.envs[0]
        
        # Ensure filtered indices are up to date
        env._update_filtered_indices()

        # Check if safe/unsafe lists are non-empty
        use_safe = len(env.safe_indices) > 0
        use_unsafe = len(env.unsafe_indices) > 0

        if not use_safe and not use_unsafe:
            raise ValueError("No examples in filtered dataset!")

        for _ in range(eval_episodes):
            # Decide which list to sample from
            if use_safe and use_unsafe:
                chosen_list = env.safe_indices if np.random.uniform() < 0.5 else env.unsafe_indices
            elif use_safe:
                chosen_list = env.safe_indices
            elif use_unsafe:
                chosen_list = env.unsafe_indices
            else:
                raise ValueError("No valid indices to sample from")

            idx = np.random.choice(chosen_list)
            self.eval_indices.append(idx)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            tp = 0
            fp = 0
            total_positives = 0
            total_negatives = 0
            action_counts = {a: 0 for a in range(self.model.action_space.n)}
            label_counts = {0:0, 1:0, -1:0}

            for idx in self.eval_indices:
                # Set environment to specific index
                self.eval_env.envs[0].current_index = idx
                obs = self.eval_env.reset()
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)

                try:
                    action_counts[action.item()] += 1
                except:
                    action_counts[int(action)] += 1

                true_label = info[0].get('true_label', -1)
                action_taken = info[0].get('action_taken', -1)

                if true_label in label_counts:
                    label_counts[true_label] += 1
                else:
                    label_counts[true_label] = 1

                if true_label == 1:
                    total_positives += 1
                    if action_taken != 0:
                        tp += 1
                else:
                    total_negatives += 1
                    if action_taken != 0:
                        fp += 1

            avg_reward = np.mean(rewards)
            tpr = tp / total_positives if total_positives > 0 else 0
            fpr = fp / total_negatives if total_negatives > 0 else 0

            print(f"\n[Eval @ Step {self.num_timesteps}]")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  TPR: {tpr:.2f}, FPR: {fpr:.2f}")
            print(f"  Action Distribution: {action_counts}")
            print(f"  Label Distribution: {label_counts}")

            with open("DEMOmetrics.txt", "a") as f:
                f.write(f"Step {self.num_timesteps}:\n")
                f.write(f"  Avg Reward: {avg_reward:.2f}\n")
                f.write(f"  TPR: {tpr:.2f}, FPR: {fpr:.2f}\n")
                f.write(f"  Action Distribution: {action_counts}\n")
                f.write(f"  Label Distribution: {label_counts}\n\n")

            # Early stopping logic (unchanged)
            self.recent_rewards.append(avg_reward)
            self.recent_tprs.append(tpr)

            if len(self.recent_rewards) > self.patience:
                reward_plateaued = (
                    max(self.recent_rewards[-self.patience:]) - min(self.recent_rewards[-self.patience:]) < self.delta
                )
                tpr_delta = self.recent_tprs[-1] - self.recent_tprs[-self.patience]
                tpr_plateaued = abs(tpr_delta) < self.delta or tpr_delta < 0

                if reward_plateaued and tpr_plateaued:
                    print("✅ Early stopping triggered (reward and TPR plateaued or declined).")
                    self.model.stop_training = True
                    return False

        return True


# --- Load Dataset and Create Train/Test Splits ---
dataset = load_from_disk('FINALCOMPLETE_DATA_SPLIT')
split = dataset.train_test_split(test_size=0.2)
train_set = split['train']
test_set = split['test']

# --- Curriculum Learning Setup ---
curriculum_steps = [0, 50_000, 100_000, 150_000]
difficulties = ['easy', 'medium', 'hard']
# Note difficulties has 3 elements, curriculum_steps has 4 now to cover all phases.

total_timesteps = 150_000

# --- Initialize Environments ---
env = JailbreakDefenseEnv(train_set, difficulty='easy')
eval_env = JailbreakDefenseEnv(test_set, difficulty='easy')

# --- Initialize Model ---
model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.03)

# --- Training Loop with Curriculum + Evaluation ---
# --- Training Loop with Curriculum + Evaluation ---
for i in range(1, len(curriculum_steps)):
    next_step = curriculum_steps[i]
    prev_step = curriculum_steps[i-1]
    timesteps_to_learn = next_step - prev_step

    print(f"\n🔄 Starting phase {i}: difficulty='{difficulties[i-1]}', timesteps={timesteps_to_learn}")

    # === Recreate environments with new difficulty ===
    env = DummyVecEnv([make_env(train_set, difficulties[i-1])])
    eval_env = DummyVecEnv([make_env(test_set, difficulties[i-1])])

    # Assign new environment to model
    model.env = env

    # Create callback with new eval_env
    eval_callback = EvaluationCallback(eval_env=eval_env, eval_freq=5000)

    # Train
    model.learn(
        total_timesteps=timesteps_to_learn,
        reset_num_timesteps=(i == 1),  # Only reset on first phase
        callback=eval_callback
    )

# --- Save the Trained Model ---
model.save("DEMO")
