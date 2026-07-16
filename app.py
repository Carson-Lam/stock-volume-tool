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

    include_outliers = st.checkbox(
        "Include outliers",
        value=True,
        help=(
            "If unchecked, unusually extreme-volume bars are excluded before "
            "computing the rising/falling averages."
        ),
    )
    if not include_outliers:
        outlier_sd = st.number_input(
            "Outlier threshold (standard deviations)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help=(
                "A bar counts as an outlier if its volume is more than this "
                "many standard deviations from the average."
            ),
        )
    else:
        outlier_sd = None

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

def mean_excluding_outliers(volumes: pd.Series, sd_threshold: float | None):
    if volumes.empty:
        return float("nan"), 0

    if sd_threshold is None:
        return volumes.mean(), 0

    mean1 = volumes.mean()
    std1 = volumes.std()

    if std1 == 0 or pd.isna(std1):
        return mean1, 0

    z_scores = (volumes - mean1).abs() / std1
    keep_mask = z_scores <= sd_threshold
    filtered = volumes[keep_mask]

    if filtered.empty:
        return mean1, 0

    return filtered.mean(), int((~keep_mask).sum())

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

    rising_avg, rising_excluded = mean_excluding_outliers(
        df.loc[rising_mask, "Volume"], outlier_sd
    )
    falling_avg, falling_excluded = mean_excluding_outliers(
        df.loc[falling_mask, "Volume"], outlier_sd
    )

    rising_help = "Average volume on bars that closed above their own open (green bars)."
    if outlier_sd is not None:
        rising_help += f" {rising_excluded} outlier bar(s) excluded (>{outlier_sd} SD)."

    falling_help = "Average volume on bars that closed below their own open (red bars)."
    if outlier_sd is not None:
        falling_help += f" {falling_excluded} outlier bar(s) excluded (>{outlier_sd} SD)."

    latest_is_rising = latest_row["Close"] > latest_row["Open"]
    latest_is_falling = latest_row["Close"] < latest_row["Open"]
    if latest_is_rising:
        latest_triangle, latest_color = "▲", "#2ada5c"   
    elif latest_is_falling:
        latest_triangle, latest_color = "▼", "#d1242f"   
    else:
        latest_triangle, latest_color = None, None      
    
    # def render_metric(col, label, value_str, triangle=None, color=None, tooltip=None):
    #     triangle_html = (
    #         f" <span style='margin-left:2px;color:{color};font-size:2.0rem;vertical-align:middle;'>{triangle}</span>"
    #         if triangle
    #         else ""
    #     )
    #     title_attr = f" title='{tooltip}'" if tooltip else ""
    #     col.markdown(
    #         f"""
    #         <div{title_attr}>
    #             <div style='font-size:0.875rem;color:#808495;margin-bottom:2px;'>{label}</div>
    #             <div style='font-size:1.75rem;line-height:1.2;margin-bottom:16px;'>{value_str}{triangle_html}
    #             </div>
    #         </div>
    #         """,
    #         unsafe_allow_html=True,
    #     )
    def render_metric(col, label, value_str, triangle=None, color=None, tooltip=None):
        with col:
            st.metric(label, value="\u00A0", help=tooltip) 

            triangle_html = (
                f" <span style='margin-left:2px;color:{color};font-size:2.0rem;vertical-align:middle;'>{triangle}</span>"
                if triangle else ""
            )
            st.markdown(
                f"""
                <div style='margin-top:-5rem;font-size:2.2rem;'>
                    {value_str}{triangle_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader(f"{ticker_input}: Volume Summary")
    c1, c2, c3, c4 = st.columns(4)
    render_metric(
        c1,       
        "Today's volume",
        f"{latest_volume:,}",
        triangle=latest_triangle,
        color=latest_color,
        tooltip=str(latest_date),
    )
    render_metric(
        c2,
        "Avg volume: Rising",
        f"{int(rising_avg):,}" if pd.notna(rising_avg) else "N/A",
        triangle="▲",
        color="#2ada5c",
        tooltip=rising_help,
    )
    render_metric(
        c3,
        "Avg volume: Falling",
        f"{int(falling_avg):,}" if pd.notna(falling_avg) else "N/A",
        triangle="▼",
        color="#d1242f",
        tooltip=falling_help,
    )
    render_metric(
        c4,
        "Avg volume:", f"{avg_volume:,}"
    )
    
    x_labels = df[date_col].dt.strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df["Volume"],
            marker_color="#4C82F7",
            name="Volume",
        )
    )
    fig.update_layout(
        title=f"{ticker_input} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Shares traded",
        xaxis_type="category",
        bargap=0.1,
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(nticks=15)
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