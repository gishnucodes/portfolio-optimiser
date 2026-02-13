"""Streamlit Dashboard for RL vs LLM Trading System."""

import sqlite3
from datetime import datetime, timedelta

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import DB_PATH, TICKERS
from engine.ledger import get_decisions, get_equity_curve, get_latest_summary
from evaluation.metrics import (
    calculate_cumulative_return,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_win_rate,
)

st.set_page_config(
    page_title="RL vs. LLM Trader",
    page_icon="📈",
    layout="wide",
)

st.title("🤖 RL vs. LLM: Duel of the Models")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Configuration")

# Value of one
if "refresh" not in st.session_state:
    st.session_state.refresh = 0

if st.sidebar.button("Refresh Data"):
    st.session_state.refresh += 1
    st.rerun()

# ── Data Loading ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_data():
    conn = sqlite3.connect(DB_PATH)
    
    # Decisions
    decisions_df = pd.read_sql_query(
        "SELECT * FROM decisions ORDER BY date DESC, window DESC",
        conn,
        parse_dates=["date"],
    )
    
    # Daily Summary
    summary_df = pd.read_sql_query(
        "SELECT * FROM daily_summary ORDER BY date ASC",
        conn,
        parse_dates=["date"],
    )
    
    conn.close()
    return decisions_df, summary_df


try:
    decisions, summary = load_data()
except Exception as e:
    st.error(f"Failed to load data from {DB_PATH}: {e}")
    st.stop()
    
if decisions.empty or summary.empty:
    st.warning("No data found in ledger. Run a backtest or live mode to generate data.")
    st.stop()

# ── Sidebar Inputs ───────────────────────────────────────────────────────────

# Date Range
min_date = summary["date"].min().date()
max_date = summary["date"].max().date()

col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
end = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

# Ticker Filter
selected_tickers = st.sidebar.multiselect("Tickers", TICKERS, default=TICKERS)

# Filter by date
mask_dec = (decisions["date"].dt.date >= start) & (decisions["date"].dt.date <= end)
decisions_filtered = decisions[mask_dec]

mask_sum = (summary["date"].dt.date >= start) & (summary["date"].dt.date <= end)
summary_filtered = summary[mask_sum]

if summary_filtered.empty:
    st.warning(f"No data found for the selected range ({start} to {end}).")
    st.stop()
    
# Use filtered data moving forward
decisions = decisions_filtered
summary = summary_filtered.drop_duplicates(subset=["date", "agent"], keep="last")

# ── Metrics Header ───────────────────────────────────────────────────────────

def get_agent_metrics(agent_key):
    agent_summary = summary[summary["agent"] == agent_key]
    if agent_summary.empty:
        return None
        
    equity_curve = agent_summary["equity"]
    daily_returns = equity_curve.pct_change().dropna()
    daily_pnl = agent_summary["pnl"].diff().dropna()
    total_ret = calculate_cumulative_return(equity_curve)
    
    return {
        "Total Return": f"{total_ret:.2%}",
        "Current Equity": f"${equity_curve.iloc[-1]:,.2f}",
        "Sharpe": f"{calculate_sharpe(daily_returns):.2f}",
        "Max Drawdown": f"{calculate_max_drawdown(equity_curve):.2%}",
        "Win Rate": f"{calculate_win_rate(daily_pnl):.1%}",
    }

st.subheader("🏆 Scoreboard")

col_rl, col_llm = st.columns(2)

rl_metrics = get_agent_metrics("rl")
if rl_metrics:
    with col_rl:
        st.markdown("### 🧠 RL Agent (PPO)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", rl_metrics["Total Return"])
        c2.metric("Equity", rl_metrics["Current Equity"])
        c3.metric("Sharpe", rl_metrics["Sharpe"])
        c4, c5 = st.columns(2)
        c4.metric("Max Drawdown", rl_metrics["Max Drawdown"])
        c5.metric("Win Rate", rl_metrics["Win Rate"])

llm_metrics = get_agent_metrics("llm")
if llm_metrics:
    with col_llm:
        st.markdown("### 💬 LLM Agent (Llama 3.3)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", llm_metrics["Total Return"])
        c2.metric("Equity", llm_metrics["Current Equity"])
        c3.metric("Sharpe", llm_metrics["Sharpe"])
        c4, c5 = st.columns(2)
        c4.metric("Max Drawdown", llm_metrics["Max Drawdown"])
        c5.metric("Win Rate", llm_metrics["Win Rate"])

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📈 Equity Curves", "📜 Trade Log", "📊 Analysis"])

with tab1:
    st.subheader("Performance Over Time")
    
    # Reshape summary for plotting
    pivot_equity = summary.pivot(index="date", columns="agent", values="equity")
    pivot_equity = pivot_equity.reset_index()
    
    fig = px.line(
        pivot_equity, 
        x="date", 
        y=["rl", "llm"], 
        title="Portfolio Equity ($)",
        labels={"value": "Equity ($)", "date": "Date", "variable": "Agent"},
        color_discrete_map={"rl": "#636EFA", "llm": "#EF553B"}
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Decision Log")
    
    # Search Filter
    search = st.text_input("Search Ticker or Reasoning", "")
    if search:
        decisions = decisions[
            decisions["ticker"].str.contains(search, case=False) |
            decisions["reasoning"].fillna("").str.contains(search, case=False)
        ]
        
    # Styling
    def highlight_action(val):
        color = 'grey'
        if val == 'BUY':
            color = 'green'
        elif val == 'SELL':
            color = 'red'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        decisions[[
            "date", "window", "ticker", "agent", "action", "price", "confidence", "reasoning"
        ]].style.applymap(highlight_action, subset=["action"]),
        use_container_width=True,
        height=500
    )

with tab3:
    st.subheader("Detailed Analysis")
    st.write("Coming soon: Alpha vs Benchmark, Sector Analysis.")
    
    # Example: Action distribution
    st.markdown("#### Action Distribution")
    dist = decisions.groupby(["agent", "action"]).size().reset_index(name="count")
    fig = px.bar(
        dist, x="agent", y="count", color="action", barmode="group",
        color_discrete_map={"BUY": "green", "SELL": "red", "HOLD": "grey"}
    )
    st.plotly_chart(fig, use_container_width=True)
