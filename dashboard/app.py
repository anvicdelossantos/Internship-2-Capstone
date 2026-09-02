"""
app.py — Real-Time Dashboard (DAY 5 DELIVERABLE)
--------------------------------------------------
A Streamlit dashboard that reads risk logs directly from the same SQLite
database the FastAPI service writes to. Streamlit was picked (per the
plan's recommendation) because it needs almost no frontend code to get
a working, auto-refreshing UI.

Run with:
  streamlit run app.py
Then open the URL it prints (usually http://localhost:8501).

NOTE: This reads the DB file directly rather than calling the API's
/logs endpoint — either approach works. Reading the DB directly is
simpler for a capstone demo; calling the API is more "correct"
microservice architecture if you want to extend this later.
"""

import streamlit as st
import pandas as pd
import sqlite3
import time
import os

st.set_page_config(page_title="Logistics Risk Dashboard", layout="wide")

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "api", "logistics.db")


def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=["id", "driver_id", "risk_level", "risk_score", "source", "timestamp"])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM risk_logs ORDER BY id DESC", conn)
    conn.close()
    return df


st.title("🚚 Smart AI Logistics Intelligence Dashboard")
st.caption("Live driver risk monitoring — data refreshes automatically every 5 seconds")

df = load_data()

if df.empty:
    st.info("No data yet. Send some requests to the FastAPI /log_data endpoint to see results here.")
else:
    # ---- Top metric row ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Logged Trips", len(df))
    col2.metric("High Risk", int((df["risk_level"] == "high").sum()))
    col3.metric("Medium Risk", int((df["risk_level"] == "medium").sum()))
    col4.metric("Avg Risk Score", f"{df['risk_score'].mean():.1f}")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("📋 Live Data Table")

        def color_risk(val):
            colors = {"low": "background-color: #d4edda",
                      "medium": "background-color: #fff3cd",
                      "high": "background-color: #f8d7da"}
            return colors.get(val, "")

        styled = df.style.map(color_risk, subset=["risk_level"])
        st.dataframe(styled, use_container_width=True, height=400)

    with right:
        st.subheader("🚦 Risk Indicators")
        counts = df["risk_level"].value_counts()
        for level, color in [("high", "🔴"), ("medium", "🟡"), ("low", "🟢")]:
            n = counts.get(level, 0)
            st.markdown(f"{color} **{level.title()}**: {n} drivers")

    st.divider()
    st.subheader("📊 Analytics")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.bar_chart(df["risk_level"].value_counts())
    with chart_col2:
        st.line_chart(df.sort_values("id")["risk_score"])

# Simple auto-refresh loop
st.button("🔄 Refresh now")
time.sleep(5)
st.rerun()
