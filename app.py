"""
Stock Volume Tracker
---------------------
A simple, free tool: enter a ticker, pick a time period, and see how many
shares were traded (volume) over that period.

Data source: Yahoo Finance, via the free/open-source `yfinance` library
(no API key required).

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stock Volume Tracker", layout="wide")

st.title("Stock Volume Tracker")
st.caption(
    "Enter a ticker and a time period to see how many shares were traded "
    "over that time. Data from Yahoo Finance."
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    ticker_input = st.text_input(
        "Ticker symbol", value="AAPL", help="e.g. AAPL, MSFT, TSLA, NVDA"
    ).strip().upper()

    range_mode = st.radio(
        "Time period",
        ["Quick range", "Custom dates"],
        horizontal=True,
    )

    if range_mode == "Quick range":
        quick_choice = st.selectbox(
            "Select range",
            ["5d", "1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "max"],
            index=2,
        )
        start_date = None
        end_date = None
    else:
        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)
        start_date = st.date_input("Start date", value=default_start, max_value=today)
        end_date = st.date_input("End date", value=today, max_value=today)
        quick_choice = None

    interval = st.selectbox(
        "Bar interval",
        ["1d", "1wk", "1mo"],
        index=0,
        help="How the volume is bucketed (daily, weekly, monthly).",
    )

    fetch_clicked = st.button("Get data", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str | None, start, end, interval: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    if period:
        df = tk.history(period=period, interval=interval)
    else:
        df = tk.history(start=start, end=end, interval=interval)
    return df


def render():
    if not ticker_input:
        st.info("Enter a ticker symbol in the sidebar to get started.")
        return

    if range_mode == "Custom dates" and start_date and end_date and start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner(f"Fetching data for {ticker_input}..."):
        try:
            df = fetch_history(
                ticker_input,
                quick_choice,
                start_date,
                end_date + dt.timedelta(days=1) if end_date else None,
                interval,
            )
        except Exception as e:
            st.error(f"Couldn't fetch data for '{ticker_input}': {e}")
            return

    if df is None or df.empty:
        st.warning(
            f"No data found for '{ticker_input}' over that period. "
            "Double check the ticker symbol or try a different range."
        )
        return

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    avg_volume = int(df["Volume"].mean())

    latest_row = df.iloc[-1]
    latest_volume = int(latest_row["Volume"])
    latest_date = latest_row[date_col].date()

    # Direction of each bar: rising = that bar's Close finished above its own
    # Open, falling = Close finished below its own Open. 
    # (RARE) Bars where Close == Open are excluded from both.
    rising_mask = df["Close"] > df["Open"]
    falling_mask = df["Close"] < df["Open"]

    rising_avg = df.loc[rising_mask, "Volume"].mean()
    falling_avg = df.loc[falling_mask, "Volume"].mean()

    st.subheader(f"{ticker_input}: Volume Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Most recent volume", f"{latest_volume:,}", help=str(latest_date))
    c2.metric("Average volume: overall", f"{avg_volume:,}")
    c3.metric(
        "Avg volume: rising",
        f"{int(rising_avg):,}" if pd.notna(rising_avg) else "N/A",
        help="Average volume on bars where price closed higher than their own open",
    )
    c4.metric(
        "Avg volume: falling",
        f"{int(falling_avg):,}" if pd.notna(falling_avg) else "N/A",
        help="Average volume on bars where price closed below their own open",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df["Volume"],
            marker_color="#4C82F7",
            name="Volume",
        )
    )
    fig.update_layout(
        title=f"{ticker_input} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Shares traded",
        bargap=0.1,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show raw data"):
        st.dataframe(
            df[[date_col, "Volume", "Close"]].sort_values(date_col, ascending=False),
            use_container_width=True,
        )
        csv = df[[date_col, "Volume", "Close"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name=f"{ticker_input}_volume.csv",
            mime="text/csv",
        )


render()

st.divider()
st.caption(
    "By Carson Lam"
)