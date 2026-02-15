"""Streamlit Dashboard for RL vs LLM Trading System."""

import sqlite3
from datetime import datetime, timedelta
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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

# ── Data Processing ──────────────────────────────────────────────────────────

# Split by Mode
if "mode" not in decisions.columns:
    decisions["mode"] = "backtest"
if "mode" not in summary.columns:
    summary["mode"] = "backtest"

decisions_live = decisions[decisions["mode"] == "live"]
summary_live = summary[summary["mode"] == "live"]

decisions_bt = decisions[decisions["mode"] == "backtest"]
summary_bt = summary[summary["mode"] == "backtest"]

# Standard Backtest Filtering
mask_dec = (decisions_bt["date"].dt.date >= start) & (decisions_bt["date"].dt.date <= end)
decisions_filtered = decisions_bt[mask_dec]

mask_sum = (summary_bt["date"].dt.date >= start) & (summary_bt["date"].dt.date <= end)
summary_filtered = summary_bt[mask_sum]

# Use filtered backtest data for main tabs
decisions_main = decisions_filtered
summary_main = summary_filtered.drop_duplicates(subset=["date", "agent"], keep="last")

# ── Metrics Helper ──────────────────────────────────────────────────────────

def get_metrics_df(summary_df):
    if summary_df.empty: return pd.DataFrame()
    
    agents = summary_df["agent"].unique()
    metrics = []
    
    for agent in agents:
        agent_data = summary_df[summary_df["agent"] == agent]
        equity = agent_data["equity"]
        if equity.empty: continue
            
        metrics.append({
            "Agent": agent,
            "Total Return": calculate_cumulative_return(equity),
            "Final Equity": equity.iloc[-1],
            "Sharpe": calculate_sharpe(equity.pct_change().dropna()),
            "Max DD": calculate_max_drawdown(equity),
        })
    return pd.DataFrame(metrics).sort_values("Total Return", ascending=False)

# ── Backtest Scoreboard ──────────────────────────────────────────────────────

# ── Live Scoreboard (Primary) ────────────────────────────────────────────────

if not summary_live.empty:
    st.header("🔴 Live Status")
    
    col_live_metrics, col_live_chart = st.columns([1, 1.5])
    
    with col_live_metrics:
        st.subheader("Leaderboard")
        live_metrics = get_metrics_df(summary_live)
        if not live_metrics.empty:
             st.dataframe(
                live_metrics,
                column_config={
                    "Total Return": st.column_config.ProgressColumn("Return", format="%.2f%%", min_value=-0.1, max_value=0.1),
                    "Final Equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
                },
                hide_index=True,
                use_container_width=True
            )
        latest_date = summary_live["date"].max().date()
        st.caption(f"Last Updated: {latest_date}")

    with col_live_chart:
        st.subheader("Performance")
        live_pivot = summary_live.pivot(index="date", columns="agent", values="equity")
        fig_live = px.line(live_pivot, markers=True)
        fig_live.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=200,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_live, use_container_width=True)
        
    st.divider()

# ── Backtest Scoreboard ──────────────────────────────────────────────────────

st.subheader("⚔️ Backtest Results")
st.caption(f"Period: {start} to {end}")

bt_metrics = get_metrics_df(summary_main)
if not bt_metrics.empty:
    st.dataframe(
        bt_metrics,
        column_config={
            "Total Return": st.column_config.ProgressColumn("Return", format="%.2f%%", min_value=-0.5, max_value=0.5),
            "Final Equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
            "Sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "Max DD": st.column_config.NumberColumn("Max DD", format="%.2f%%"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    winner = bt_metrics.iloc[0]
    st.success(f"🏆 **Backtest Winner:** {winner['Agent']} ({winner['Total Return']:.2%} Return)")

else:
    st.info("No backtest data for selected range.")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Equity Curves", "📜 Trade Log", "📊 Analysis", "ℹ️ Details", "🧪 Live Experiments"])

with tab1:
    st.subheader("Performance Over Time")
    if not summary_main.empty:
        pivot_equity = summary_main.pivot(index="date", columns="agent", values="equity")
        fig = px.line(pivot_equity, title="Portfolio Equity ($)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data to plot.")

with tab2:
    st.subheader("Decision Log")
    if not decisions_main.empty:
        # Search Filter
        search = st.text_input("Search Ticker or Reasoning", "")
        if search:
            decisions_main = decisions_main[
                decisions_main["ticker"].str.contains(search, case=False) |
                decisions_main["reasoning"].fillna("").str.contains(search, case=False)
            ]
            
        def highlight_action(val):
            color = 'grey'
            if val == 'BUY': color = 'green'
            elif val == 'SELL': color = 'red'
            return f'color: {color}; font-weight: bold'

        st.dataframe(
            decisions_main[[
                "date", "window", "ticker", "agent", "action", "price", "confidence", "reasoning"
            ]].style.map(highlight_action, subset=["action"]),
            use_container_width=True,
            height=500
        )
    else:
        st.info("No decisions found for this period.")

with tab3:
    st.subheader("Detailed Analysis")
    st.info("Additional metrics coming soon.")
    if not decisions_main.empty:
        st.markdown("#### Action Distribution")
        dist = decisions_main.groupby(["agent", "action"]).size().reset_index(name="count")
        fig = px.bar(
            dist, x="agent", y="count", color="action", barmode="group",
            color_discrete_map={"BUY": "green", "SELL": "red", "HOLD": "grey"}
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Strategy Internals")
    st.markdown("""
    ### 🤖 RL Agents (PPO)
    All Reinforcement Learning agents use **Proximal Policy Optimization (PPO)** trained on 5 years of historical data. The difference lies in their **Action Constraints**:

    #### 1. PPO_STANDARD
    - **Behavior**: Pure RL. Learns to maximize risk-adjusted returns (Sharpe Ratio).
    - **Constraints**: None.
    - **Typical Style**: Buy & Hold (in bull markets) or steady accumulation.

    #### 2. PPO_DIP_BUYER
    - **Behavior**: Conservative. Prioritizes cash preservation.
    - **Constraints**:
        - **Cash Rule**: Must maintain at least **$2,000** (20%) in cash.
        - **Buy Rule**: Can only dip below $2,000 cash if **RSI < 30** (Oversold).
    - **Goal**: Buy cheap, avoid buying tops.

    #### 3. PPO_MOMENTUM
    - **Behavior**: Aggressive trend follower.
    - **Constraints**:
        - **Trend Filter**: Only BUY if **Price > 50-day EMA**.
        - **Stop Loss**: SELL/HOLD if **Price < 50-day EMA**.
    - **Goal**: Ride strong uptrends, exit quickly when trend breaks.

    ---

    ### 💬 LLM Agent (Llama 3.3)
    - **Engine**: Llama 3.3 via Ollama.
    - **Logic**: 
        1. Receives recent price data + technical indicators (RSI, MACD, Bollinger Bands).
        2. Receives a text summary of market structure.
        3. Reasons about the setup in natural language.
        4. Outputs a structured JSON decision (`BUY`/`SELL`/`HOLD`).
    - **Style**: Context-aware, can explain its reasoning.
    """)

with tab5:
    st.header("🧪 Live Experiments")
    st.markdown("Tracking models currently deployed in `run_live.py`. Data is updated daily via `run_live.py`.")
    
    if summary_live.empty:
        st.warning("No live experiment data found yet. Run `python run_live.py` to generate data.")
    else:
        # Live Scoreboard
        live_metrics = get_metrics_df(summary_live)
        st.markdown("### 📡 Live Status")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(
                live_metrics,
                column_config={
                    "Total Return": st.column_config.ProgressColumn("Return", format="%.2f%%", min_value=-0.2, max_value=0.2),
                },
                hide_index=True,
                use_container_width=True
            )
            
        with col2:
            latest_date = summary_live["date"].max()
            st.metric("Last Update", f"{latest_date}")
            
        # Live Charts
        st.subheader("Live Performance")
        live_pivot = summary_live.pivot(index="date", columns="agent", values="equity")
        fig_live = px.line(live_pivot, title="Live Equity ($)", markers=True)
        st.plotly_chart(fig_live, use_container_width=True)

        st.subheader("Recent Activity")
        st.dataframe(decisions_live.sort_values("date", ascending=False).head(50), use_container_width=True)
