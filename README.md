# Prompt Defender RL: Adaptive LLM Jailbreak Defense

![System Diagram](diagram.png)

![Confusion Matrix with Stats](confusion_matrices_with_stats.png)

---

## Project Overview & Motivation

- Deploying new LLMs with static wrappers takes a lot of time.
- This creates a vulnerability period where the system is exposed to attacks.
- We need a temporary, dynamic workaround to secure models during this critical phase.
- **Objective:** Create an adaptive defense layer that learns to detect and recover from LLM jailbreaks in real time using deep reinforcement learning (RL).
- **Goal:** To outperform static wrappers and showcase an autonomous, self-healing defense prototype.

### Agent State Vector
- Prompt Embeddings: Sentence-transformer embeddings of the prompt.
- Response Information: Candidate response embeddings and policy violation logits.
- Heuristics: Toxicity scores and jailbreak patterns.
- Wrapper Metadata: Guardrail flags, latency, etc.

### Datasets
- Synthetic and benchmark jailbreak streams.
- JailbreakBench & AdvPromptBench: Primary data sources.
- Benign Queries: Mixed in harmless queries to measure the false-positive rate.

### Training Loop
- Curriculum learning (starting with simple attacks) and early stopping on reward plateaus.

---

# Setup Instructions

Follow these steps to run the full Prompt Defender RL stack:

## 1. Start the FastAPI Backend

Run the backend API using Uvicorn:

```bash
cd modifiedrun/
uvicorn app:app --reload
```

## 2. Start the Frontend (React)

Run the frontend development server:

```bash
cd prompt-defender-rl/
npm run dev
```

## 3. Load Sample Prompts

Run the sample prompt loader script to simulate quick responses to multiple prompts:

```bash
cd modifiedrun/
python3 load_sample_prompts.py
```

---

**Order does not matter, but the backend should be running before using the frontend.**

---

## Next Steps

- **Benchmark Report:** Generate a report comparing our metrics against existing static wrappers (e.g., Rebuff, PromptGuard).
- **Ablation Study:** Analyze the impact of different features on performance.
- **Self-Healing:** Deployment on ONNX is solid, next steps would be to receive feedback and continue adapting the environment.
