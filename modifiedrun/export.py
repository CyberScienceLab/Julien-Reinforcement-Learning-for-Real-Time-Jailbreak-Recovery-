from stable_baselines3 import PPO
from datasets import load_from_disk
import numpy as np
import torch

# Load dataset and model
dataset = load_from_disk('COMPLETE_DATA')
#model = PPO.load("ModelTEST")
model = PPO.load("DEMO")

policy = model.policy
policy.eval()

obs_shape = model.observation_space.shape  # e.g., (4,)
device = next(policy.parameters()).device  # automatically get policy's device
dummy_input = torch.randn(1, *obs_shape, dtype=torch.float32).to(device)

# Trace the deterministic=False path explicitly
class WrappedPolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.extractor = policy.features_extractor
        self.mlp = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, x):
        features = self.extractor(x)
        latent_pi, _ = self.mlp(features)
        action_logits = self.action_net(latent_pi)
        return action_logits  # Raw logits (used for sampling or argmax)



wrapped_policy = WrappedPolicy(policy).to(device)
torch.onnx.export(
    wrapped_policy,
    dummy_input,
    "DEMO.onnx",
    input_names=["input"],
    output_names=["action"],
    dynamic_axes={"input": {0: "batch_size"}},
    opset_version=11
)

