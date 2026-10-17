import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="TrustRAG Observability", layout="wide")
st.title("TrustRAG — Observability Dashboard")

LOG_PATH = Path("data") / "interaction_log.csv"

if not LOG_PATH.exists():
    st.warning("No interactions logged yet. Ask a few questions through the app first, then refresh.")
else:
    df = pd.read_csv(LOG_PATH, parse_dates=["timestamp"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total queries", len(df))
    col2.metric("Avg latency (s)", f"{df['latency_seconds'].mean():.2f}")
    col3.metric("Grounding rate", f"{100 * df['grounded'].mean():.0f}%")
    col4.metric("Avg sources/answer", f"{df['num_sources'].mean():.1f}")

    st.subheader("Latency over time")
    st.line_chart(df.set_index("timestamp")["latency_seconds"])

    st.subheader("Queries per domain")
    st.bar_chart(df["domain"].value_counts())

    st.subheader("Grounding rate by domain")
    grounding_by_domain = df.groupby("domain")["grounded"].mean() * 100
    st.bar_chart(grounding_by_domain)

    st.subheader("Raw log")
    st.dataframe(df.sort_values("timestamp", ascending=False))
