# Progress_Julien Setup Instructions

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
