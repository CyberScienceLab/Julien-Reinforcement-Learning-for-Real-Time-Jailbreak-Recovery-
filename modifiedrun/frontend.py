import streamlit as st
import requests
import ollama

API_URL = "http://localhost:8000/predict"
LLM_URL = "http://localhost:8000/llm_response"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

from sentence_transformers import SentenceTransformer
import numpy as np


def generate(prompt: str, model='mistral') -> str:
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']

st.title("Prompt Defender RL Demo")

prompt = st.text_area("Enter your prompt:")

if 'embedder' not in st.session_state:
    st.session_state['embedder'] = SentenceTransformer(EMBEDDING_MODEL)

if st.button("Submit"):
    embedder = st.session_state['embedder']
    prompt_embedding = embedder.encode(prompt).astype(float)
    obs = prompt_embedding.tolist()  # Use only prompt embedding for now
    OBS_LENGTH = 389
    if len(obs) < OBS_LENGTH:
        obs += [0.0] * (OBS_LENGTH - len(obs))
    payload = {"obs": obs}
    response = requests.post(API_URL, json=payload)
    if response.status_code == 200:
        action = response.json()["action"]
        action_names = ["ALLOW", "BLOCK", "REWRITE", "ASK", "ESCALATE"]
        st.session_state['last_action'] = action
        st.session_state['last_prompt'] = prompt
        st.write(f"**Action:** {action_names[action]}")
        # If not ALLOW, get LLM response
        if action != 0:
            llm_payload = {"prompt": prompt, "action_id": action}
            llm_response = requests.post(LLM_URL, json=llm_payload)
            if llm_response.status_code == 200:
                st.session_state['last_llm_response'] = llm_response.json()['response']
                st.write(f"**LLM Response:** {st.session_state['last_llm_response']}")
            else:
                st.write("LLM response error.")
    else:
        st.write("Prediction error.")

# Show moderator UI if last action was ESCALATE
if st.session_state.get('last_action', None) == 4:
    st.write(f"**Action:** ESCALATE")
    if 'last_llm_response' in st.session_state:
        st.write(f"**LLM Response:** {st.session_state['last_llm_response']}")
    user_decision = st.radio("Moderator decision:", ["Approve", "Decline"], key="mod_decision")
    if st.button("Submit Moderator Decision"):
        st.write(f"Moderator decision: {user_decision}")
        # Optionally clear state after submission
        del st.session_state['last_action']
        if 'last_llm_response' in st.session_state:
            del st.session_state['last_llm_response']
