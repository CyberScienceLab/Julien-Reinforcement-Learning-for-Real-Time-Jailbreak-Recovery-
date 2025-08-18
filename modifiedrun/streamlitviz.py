import streamlit as st
import requests
import pandas as pd
from streamlit_autorefresh import st_autorefresh

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="RL Defense Metrics", layout="wide")

st.title("RL Defense Metrics Dashboard")

# Fetch inference history
def fetch_history():
    try:
        resp = requests.get(f"{API_BASE}/inference/history")
        data = resp.json()
        return pd.DataFrame(data["requests"])
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return pd.DataFrame()

# Fetch metrics
def fetch_metrics():
    try:
        resp = requests.get(f"{API_BASE}/metrics")
        return resp.json()
    except Exception as e:
        st.error(f"Error fetching metrics: {e}")
        return {}

# Sidebar refresh
refresh_interval = st.sidebar.slider("Refresh interval (sec)", 2, 30, 5)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

# Auto-refresh using streamlit_autorefresh
if auto_refresh:
    st_autorefresh(interval=refresh_interval * 1000, key="dashboard_autorefresh")

# Main dashboard
history_df = fetch_history()
metrics = fetch_metrics()

col1, col2, col3 = st.columns(3)

# Reward curve (simulate with latency as reward proxy)
with col1:
    st.subheader("Reward Curve (Latency as Proxy)")
    if not history_df.empty:
        history_df["step"] = range(1, len(history_df)+1)
        st.line_chart(history_df.set_index("step")["latency"])
    else:
        st.info("No data yet.")

# Attack Success (show % of BLOCK/ESCALATE actions)
with col2:
    st.subheader("Attack Success (High Threat Actions)")
    if metrics:
        st.metric("Threat Level", metrics.get("threatLevel", "N/A"))
        st.metric("High Threat %", 
                  f"{(metrics['actionCounts']['BLOCK'] + metrics['actionCounts']['ESCALATE']) / max(metrics['totalRequests'],1) * 100:.1f}%")
        st.bar_chart({k: v for k, v in metrics["actionCounts"].items() if k in ["BLOCK", "ESCALATE"]})
    else:
        st.info("No metrics yet.")

# Action Breakdown
with col3:
    st.subheader("Action Breakdown")
    if metrics:
        st.bar_chart(metrics["actionCounts"])
    else:
        st.info("No metrics yet.")

# Show full inference log
st.subheader("Recent Inference Log")
if not history_df.empty:
    st.dataframe(history_df[["timestamp", "prompt", "action", "latency", "confidence", "cached"]].sort_values("timestamp", ascending=False))
else:
    st.info("No inference records yet.")