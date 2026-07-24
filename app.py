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
from streamlit_autorefresh import st_autorefresh

st_autorefresh(interval=10_000, key="minute_data_refresh") 

st.set_page_config(page_title="Stock Volume Tracker", layout="wide")


st.title("Stock Volume Average")
st.caption(
    "Enter a ticker and a time period to see how many shares were traded "
    "over that time. Hover over the average volumes to see the top 2 highest-volume bars."
)

# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(ticker: str, period: str | None, start, end, interval: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        if period:
            df = tk.history(period=period, interval=interval)
        else:
            df = tk.history(start=start, end=end, interval=interval)
    except Exception:
        return pd.DataFrame()
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_quote(ticker: str):
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    return {
        "price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "change": info.get("regularMarketChange"),
        "pct_change": info.get("regularMarketChangePercent"),
        "market_state": info.get("marketState"),
        "market_time": info.get("regularMarketTime"),
        "currency": info.get("currency", "USD"),
        "long_name": info.get("longName") or info.get("shortName"),
        "exchange": info.get("fullExchangeName") or info.get("exchange"),
    }

@st.cache_data(ttl=120, show_spinner=False)
def fetch_latest_minute(ticker: str, minute_key: str):
    tk = yf.Ticker(ticker)
    try:
        recent = tk.history(period="1d", interval="1m")
    except Exception:
        return None
    recent = recent[recent["Volume"] > 0].tail(2) 
    if len(recent) < 2:
        return None
    recent = recent.reset_index()
    time_col = "Datetime" if "Datetime" in recent.columns else recent.columns[0]
    recent = recent.rename(columns={time_col: "Datetime"})
    prev_vol, latest_vol = recent["Volume"].iloc[0], recent["Volume"].iloc[1]
    pct = ((latest_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
    latest_t = recent["Datetime"].iloc[1]
    end_t = latest_t + pd.Timedelta(minutes=1)
    time_label = f"{latest_t.hour}:{latest_t.minute:02d}-{end_t.hour}:{end_t.minute:02d}"
    return {"pct": pct, "time_label": time_label}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_minute_volume(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    try:
        mdf = tk.history(period="7d", interval="1m")
    except Exception:
        return pd.DataFrame()
    if mdf.empty:
        return mdf
    mdf = mdf[mdf["Volume"] > 0]
    if mdf.empty:
        return mdf
    mdf = mdf.reset_index()
    time_col = "Datetime" if "Datetime" in mdf.columns else mdf.columns[0]
    mdf = mdf.rename(columns={time_col: "Datetime"})
    mdf["Day"] = mdf["Datetime"].dt.date
    mdf["PrevDatetime"] = mdf.groupby("Day")["Datetime"].shift(1)
    mdf["VolPctChange"] = mdf.groupby("Day")["Volume"].pct_change() * 100

    def fmt_time(t):
        return f"{t.hour}:{t.minute:02d}"

    mdf["TimeLabel"] = mdf.apply(
        lambda r: f"{fmt_time(r['PrevDatetime'])}-{fmt_time(r['Datetime'])}"
        if pd.notna(r["PrevDatetime"]) else fmt_time(r["Datetime"]),
        axis=1,
    )
    mdf["DateTimeLabel"] = mdf["Day"].astype(str) + " " + mdf["TimeLabel"]
    return mdf

def current_minute_key() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

def mean_excluding_outliers(volumes: pd.Series, sd_threshold: float | None):
    if volumes.empty:
        return float("nan"), 0, pd.Series(True, index=volumes.index)

    if sd_threshold is None:
        return volumes.mean(), 0, pd.Series(True, index=volumes.index)

    mean1 = volumes.mean()
    std1 = volumes.std()

    if std1 == 0 or pd.isna(std1):
        return mean1, 0, pd.Series(True, index=volumes.index)

    z_scores = (volumes - mean1).abs() / std1
    keep_mask = z_scores <= sd_threshold
    filtered = volumes[keep_mask]

    if filtered.empty:
        return mean1, 0, pd.Series(True, index=volumes.index)

    return filtered.mean(), int((~keep_mask).sum()), keep_mask

def compute_rsi(close: pd.Series, period: int = 9) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_vr(close: pd.Series, volume: pd.Series, period: int = 26) -> pd.Series:
    change = close.diff()
    up_vol = volume.where(change > 0, 0)
    down_vol = volume.where(change < 0, 0)
    flat_vol = volume.where(change == 0, 0)
    up_sum = up_vol.rolling(period).sum()
    down_sum = down_vol.rolling(period).sum()
    flat_sum = flat_vol.rolling(period).sum()
    return (up_sum + flat_sum / 2) / (down_sum + flat_sum / 2) * 100

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif, dea

def normalize_exchange(raw_exchange: str | None) -> str:
    if not raw_exchange:
        return ""
    raw = raw_exchange.upper()
    if "NASDAQ" in raw or raw in ("NMS", "NCM", "NGM"):
        return "NASDAQ"
    if "NYSE" in raw or raw == "NYQ":
        return "NYSE"
    return raw_exchange

def render_filter_card(col, title, passed, detail_text, formula_html=None):
    with col:
        color = "#2ada5c" if passed else "#d1242f"
        bg = "rgba(42,218,92,0.08)" if passed else "rgba(209,36,47,0.08)"
        status_text = "Met" if passed else "Not met"

        if formula_html:
            title_html = f"""
            <span class="filter-tooltip" style="position:relative;cursor:help;
                border-bottom:2px dotted #808495;">
                {title}
                <div class="filter-tooltip-content" style="display:none;position:absolute;
                    top:100%;left:0;background:#262730;border:1px solid #444;
                    border-radius:8px;padding:12px 14px;font-size:0.8rem;line-height:1.6;
                    white-space:nowrap;z-index:999;margin-top:6px;font-family:monospace;
                    box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                    {formula_html}
                </div>
            </span>
            <style>.filter-tooltip:hover .filter-tooltip-content {{ display:block !important; }}</style>
            """
        else:
            title_html = title

        st.markdown(
            f"""
            <div style='border:1px solid {color}55; background:{bg}; border-radius:0.5rem;
                padding:14px 16px; border-left:2px solid {color};'>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:6px;'>
                    <span style='font-weight:600; font-size:0.95rem;'>{title_html}</span>
                </div>
                <div style='color:{color}; font-size:0.8rem; font-weight:600; margin-bottom:4px;'>{status_text}</div>
                <div style='color:#aaa; font-size:0.8rem; line-height:1.4;'>{detail_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

    if interval == "5m":
        df = df[df["Volume"] > 0]
        if df.empty:
            st.warning(f"No intraday data available for '{ticker_input}' yet today.")
            return

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else df.columns[0]

    avg_volume = int(df["Volume"].mean())
    latest_row = df.iloc[-1]
    latest_volume = int(latest_row["Volume"])
    latest_date = latest_row[date_col].date()

    quote = fetch_quote(ticker_input)
    if quote and quote["price"] is not None:
        latest_price = quote["price"]
        price_change = quote["change"]
        pct_change = quote["pct_change"]
        market_state = quote["market_state"]
    else:
        latest_price = latest_row["Close"]
        prev_close = df.iloc[-2]["Close"] if len(df) >= 2 else latest_price
        price_change = latest_price - prev_close
        pct_change = (price_change / prev_close * 100) if prev_close else 0
        market_state = "CLOSED"
    state_label = {
        "REGULAR": "Live",
        "PRE": "Pre-market",
        "POST": "After hours",
        "CLOSED": "Closed",
    }.get(market_state, "Closed")

    company_name = quote.get("long_name") if quote else None
    exchange_label = normalize_exchange(quote.get("exchange")) if quote else ""
    exchange_ticker = f"{exchange_label}: {ticker_input}" if exchange_label else ticker_input
    ticker_label = f"{company_name} ({exchange_ticker})" if company_name else exchange_ticker

    # Direction of each bar: rising = that bar's Close finished above its own
    # Open, falling = Close finished below its own Open. 
    # (RARE) Bars where Close == Open are excluded from both.
    rising_mask = df["Close"] > df["Open"]
    falling_mask = df["Close"] < df["Open"]

    rising_avg, rising_excluded, rising_keep = mean_excluding_outliers(
        df.loc[rising_mask, "Volume"], outlier_sd
    )
    falling_avg, falling_excluded, falling_keep = mean_excluding_outliers(
        df.loc[falling_mask, "Volume"], outlier_sd
    )
    excluded_idx = rising_keep[~rising_keep].index.union(falling_keep[~falling_keep].index)
    df_chart = df.drop(index=excluded_idx)

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
    
    def extreme2(mask, ascending):
        subset = df.loc[mask, [date_col, "Volume"]].sort_values("Volume", ascending=ascending).head(2)
        fmt = "%Y-%m-%d %H:%M" if interval == "5m" else "%Y-%m-%d"
        return [
            (row[date_col].strftime(fmt), f"{int(row['Volume']):,}")
            for _, row in subset.iterrows()
        ]

    rising_top2 = extreme2(rising_mask, ascending=False)
    falling_top2 = extreme2(falling_mask, ascending=True)

    def render_metric(col, label, value_str, triangle=None, color=None, tooltip=None, hover_rows=None, hover_title="Top 2 volumes"):
        with col:
            st.metric(label, value="\u00A0", help=tooltip) 

            triangle_html = (
                f" <span style='margin-left:2px;color:{color};font-size:2.0rem;vertical-align:middle;'>{triangle}</span>"
                if triangle else ""
            )

            if hover_rows:
                rows_html = "".join(
                    f"<div style='display:flex;justify-content:space-between;gap:14px;"
                    f"padding:2px 0;'><span style='color:#aaa;'>{d}</span>"
                    f"<span style='font-weight:600;'>{v}</span></div>"
                    for d, v in hover_rows
                )
                dropdown_html = f"""
                <div class="vol-tooltip" style="position:relative;display:inline;cursor:help;">
                    <span style="border-bottom:2px dotted #808495;">{value_str}</span>{triangle_html}
                    <div class="vol-tooltip-content" style="display:none;position:absolute;
                        top:100%;left:0;background:#262730;border:1px solid #444;
                        border-radius:8px;padding:10px 14px;font-size:0.85rem;
                        white-space:nowrap;z-index:999;margin-top:6px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                        <div style='color:#ddd;font-size:0.75rem;margin-bottom:4px;'>
                            {hover_title}
                        </div>
                        {rows_html}
                    </div>
                </div>

                <style>
                .vol-tooltip:hover .vol-tooltip-content {{ display:block !important; }}
                </style>
                """
            else:
                dropdown_html = f"{value_str}{triangle_html}"

            st.markdown(
                f"""
                <div style='margin-top:-5rem;font-size:2.2rem;'>
                    {dropdown_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

    price_is_up = price_change >= 0
    price_arrow = "▲" if price_is_up else "▼"
    price_color = "#2ada5c" if price_is_up else "#d1242f"
    badge_bg = "rgba(42,218,92,0.15)" if price_is_up else "rgba(209,36,47,0.15)"
    change_sign = "+" if price_is_up else "-"
    st.markdown(
        f"""
        <div style='margin-bottom:15px;'>
            <div style='font-size:1.1rem;line-height:1.2;'>{ticker_label}</div>
            <div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;line-height:1.2;'>
                <span style='font-size:2.75rem;font-weight:700;'>{latest_price:,.2f}</span>
                <span style='color:#808495;font-size:1.1rem;'>{quote.get("currency", "USD") if quote else "USD"}</span>
                <span style='background:{badge_bg};color:{price_color};padding:4px 12px;
                    border-radius:0.5rem;font-weight:600;font-size:1rem;'>
                    {price_arrow} {abs(pct_change):.2f}%
                </span>
                <span style='color:{price_color};font-size:1.1rem;'>
                    {change_sign}{abs(price_change):.2f} today
                </span>
            </div>
            <div style='color:#808495;font-size:14px;'>
                {state_label}: {latest_date.strftime('%b %d, %Y')} &middot; 
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        "Average volume: Rising",
        f"{int(rising_avg):,}" if pd.notna(rising_avg) else "N/A",
        triangle="▲",
        color="#2ada5c",
        tooltip=rising_help,
        hover_rows=rising_top2,
        hover_title="Top 2 volumes",
    )
    render_metric(
        c3,
        "Average volume: Falling",
        f"{int(falling_avg):,}" if pd.notna(falling_avg) else "N/A",
        triangle="▼",
        color="#d1242f",
        tooltip=falling_help,
        hover_rows=falling_top2,
        hover_title="Lowest 2 volumes",
    )
    render_metric(
        c4,
        "Average volume:", f"{avg_volume:,}"
    )
    
    spans_multiple_days = df_chart[date_col].dt.date.nunique() > 1
    x_labels = (
        df_chart[date_col].dt.strftime("%m-%d %H:%M" if spans_multiple_days else "%H:%M")
        if interval == "5m"
        else df_chart[date_col].dt.strftime("%Y-%m-%d")
    )

    bar_colors = [
        "#229B44" if c > o else "#9C252D" if c < o else "#888888"
        for c, o in zip(df_chart["Close"], df_chart["Open"])
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df_chart["Volume"],
            marker_color=bar_colors,
            name="Volume",
        )
    )

    fig.add_hline(y=avg_volume, line_dash="dash", line_color="#FFFFFF", line_width=3)
    fig.add_annotation(
        xref="paper", x=1.01, yref="y", y=avg_volume,
        text=" ", showarrow=False, xanchor="left",
        font=dict(color="#FFFFFF", size=11),
    )
    if show_rising_line and pd.notna(rising_avg):
        fig.add_hline(
            y=rising_avg,
            line_dash="dash",
            line_color="#2ada5c",
            line_width=3,
            layer="above"
        )
        fig.add_annotation(
            xref="paper", x=1.01, yref="y", y=rising_avg,
            text="Rising average", showarrow=False, xanchor="left",
            font=dict(color="#229B44", size=11),
        )
    if show_falling_line and pd.notna(falling_avg):
        fig.add_hline(
            y=falling_avg,
            line_dash="dash",
            line_color="#d1242f",
            line_width=3,
            layer="above"
        )
        fig.add_annotation(
            xref="paper", x=1.01, yref="y", y=falling_avg,
            text="Falling average", showarrow=False, xanchor="left",
            font=dict(color="#9C252D", size=11),
        )

    # trend_window = st.slider("Trend window (bars)", 2, 20, 5, key="volume_trend_window")
    # df["Volume_Trend"] = df["Volume"].rolling(window=trend_window, min_periods=1).mean()
    # fig.add_trace(
    #     go.Scatter(
    #         x=x_labels,
    #         y=df["Volume_Trend"],
    #         mode="lines",
    #         line=dict(color="#FFA500", width=3),
    #         name=f"{trend_window}-bar trend",
    #     )
    # )

    fig.update_layout(
        title=f"{ticker_input} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Shares traded",
        xaxis_type="category",
        bargap=0.1,
        height=500,
        margin=dict(l=10, r=90, t=50, b=10),
    )
    fig.update_xaxes(nticks=15)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show raw data"):
        st.dataframe(
            df[[date_col, "Open", "Close", "Volume"]]
            .assign(Open=lambda d: d["Open"].round(4), Close=lambda d: d["Close"].round(4))
            .sort_values(date_col, ascending=False),
            use_container_width=True,
        )
        csv = df[[date_col, "Open", "Close", "Volume"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download as CSV",
            data=csv,
            file_name=f"{ticker_input}_volume.csv",
            mime="text/csv",
        )

    st.divider()

    st.title("Technical Filter Check")
    st.caption(f"Checking {ticker_input} against 3 technical filters using daily and monthly data.")

    filter_df = fetch_history(ticker_input, "6mo", None, None, "1d")
    if filter_df is None or filter_df.empty or len(filter_df) < 70:
        st.warning("Not enough daily history to compute these indicators (need ~70 trading days minimum).")
    else:
        filter_df = filter_df.reset_index()
        rsi = compute_rsi(filter_df["Close"], period=9)
        vr = compute_vr(filter_df["Close"], filter_df["Volume"], period=26)
        dif, dea = compute_macd(filter_df["Close"])
        is_up_day = filter_df["Close"].diff() > 0
        vol_on_up_days = filter_df["Volume"].where(is_up_day)
        vol_avg_3mo_up = vol_on_up_days.rolling(63, min_periods=1).mean() 

        latest_rsi = rsi.iloc[-1]
        latest_vr = vr.iloc[-1]
        latest_dif, latest_dea = dif.iloc[-1], dea.iloc[-1]
        latest_vol = filter_df["Volume"].iloc[-1]
        latest_vol_avg_3mo_up = vol_avg_3mo_up.iloc[-1]
        rsi_above_50 = pd.notna(latest_rsi) and latest_rsi > 50

        f1_pass = rsi_above_50 and pd.notna(latest_vr) and latest_vr > 100
        f2_pass = rsi_above_50 and pd.notna(latest_dif) and pd.notna(latest_dea) and latest_dif > latest_dea
        f3_pass = (
            rsi_above_50
            and pd.notna(latest_vol_avg_3mo_up)
            and latest_vol > latest_vol_avg_3mo_up
        )

        total_passed = sum([f1_pass, f2_pass, f3_pass])

        st.markdown(f"### {total_passed} / 3 filters met")
        
        rsi_formula = (
            "<div style='color:#ddd;font-weight:600;margin-bottom:4px;'>RSI(9)</div>"
            "<div>RS = Avg Gain(9) / Avg Loss(9)</div>"
            "<div>RSI = 100 &minus; (100 / (1 + RS))</div>"
            "<div>Avg Gain(9) = (prior avg &times; 8 + today) / 9</div>"
        )
        macd_formula = (
            "<div style='color:#ddd;font-weight:600;margin-bottom:4px;'>MACD</div>"
            "<div>DIF = EMA(12) &minus; EMA(26)</div>"
            "<div>DEA = EMA(9) of DIF</div>"
        )

        fc1, fc2, fc3 = st.columns(3)
        render_filter_card(
            fc1, "Filter 1: RSI9 > 50 | VR > 100", f1_pass,
            f"RSI9: {latest_rsi:.1f} &middot; VR: {latest_vr:.1f}" if pd.notna(latest_rsi) and pd.notna(latest_vr) else "Insufficient data",
            formula_html=rsi_formula,
        )
        render_filter_card(
            fc2, "Filter 2: RSI9 > 50 | MACD DIF > DEA", f2_pass,
            f"RSI9: {latest_rsi:.1f} &middot; DIF: {latest_dif:.2f} &middot; DEA: {latest_dea:.2f}" if pd.notna(latest_rsi) and pd.notna(latest_dif) else "Insufficient data",
            formula_html=macd_formula,
        )
        render_filter_card(
            fc3, "Filter 3: RSI9 > 50 | Vol > 3mo up-avg", f3_pass,
            f"RSI9: {latest_rsi:.1f} &middot; Vol: {latest_vol:,.0f} vs 3mo up-avg {latest_vol_avg_3mo_up:,.0f}" if pd.notna(latest_rsi) and pd.notna(latest_vol_avg_3mo_up) else "Insufficient data",
        )
        
    st.divider()
    st.title("Stock Volume % Change")
    st.caption(
        "How much trading volume rose or fell from one minute to the next. "
        "Spans the last 7 trading days, updated every minute."
    )

    if mdf.empty:
        st.warning("No minute-level data available for this ticker right now.")
    else:
        day_df = (
            mdf[mdf["Day"] == selected_day]
            .dropna(subset=["VolPctChange"])
            .reset_index(drop=True)
        )

        if day_df.empty:
            st.info("Not enough minutes in this day to compute a percent change yet.")
        else:
            live = fetch_latest_minute(ticker_input, current_minute_key())
            if live:
                latest_pct = live["pct"]
                latest_time_label = live["time_label"]
            else:
                latest_pct = day_df["VolPctChange"].iloc[-1]
                latest_time_label = day_df["TimeLabel"].iloc[-1]

            latest_tri = "▲" if latest_pct > 0 else "▼" if latest_pct < 0 else None
            latest_col = "#2ada5c" if latest_pct > 0 else "#d1242f" if latest_pct < 0 else None

            if latest_pct > 0:
                badge_bg_m = "rgba(42,218,92,0.15)"
            elif latest_pct < 0:
                badge_bg_m = "rgba(209,36,47,0.15)"
            else:
                badge_bg_m = "rgba(128,132,149,0.15)"

            st.markdown(
                f"""
                <div style='margin-bottom:15px;'>
                    <div style='font-size:1.1rem;line-height:1.2;'>Most recent minute % change</div>
                    <div style='display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;line-height:1.2;'>
                        <span style='font-size:2.75rem;font-weight:700;'>{latest_time_label}</span>
                        <span style='background:{badge_bg_m};color:{latest_col or "#808495"};padding:4px 12px;
                            border-radius:0.5rem;font-weight:600;font-size:1rem;'>
                            {latest_tri or ""} {abs(latest_pct):.2f}%
                        </span>
                    </div>
                    <div style='color:#808495;font-size:14px;'>
                        {state_label}: {latest_date.strftime('%b %d, %Y')} &middot; 
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            x_vals = day_df["Datetime"]
            y_vals = day_df["VolPctChange"]

            min_fig = go.Figure(
                data=[go.Scatter(x=x_vals, y=y_vals, mode="lines", line=dict(color="#4C82F7"))],
            )
            min_fig.add_hline(y=0, line_dash="dot", line_color="#808495")
            min_fig.update_layout(
                title=f"{ticker_input} Minute Volume % Change ({selected_day.strftime('%b %d, %Y')})",
                xaxis=dict(range=[x_vals.min(), x_vals.max()], tickformat="%H:%M", title="Time"),
                yaxis=dict(title="% change vs. previous minute"),
                height=450,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(min_fig, use_container_width=True)

            with st.expander("Show minute-by-minute data (all days)"):
                for d in trading_days:
                    d_df = mdf[mdf["Day"] == d][["DateTimeLabel", "Volume", "VolPctChange"]].copy()
                    d_df = d_df.rename(columns={"DateTimeLabel": "Datetime"})
                    st.markdown(f"**{d.strftime('%A, %b %d, %Y')}**")
                    st.dataframe(d_df, use_container_width=True)
            



# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    
    st.write("")

    ticker_input = st.text_input(
        "Ticker symbol", value="AAPL", help="e.g. AAPL, MSFT, TSLA, NVDA"
    ).strip().upper()

    st.divider()

    st.subheader("Volume Average")
    range_mode = st.radio(
        "Time period",
        ["Quick range", "Custom dates"],
        horizontal=True,
    )

    interval = st.selectbox(
        "Bar interval",
        ["5m", "1d", "1wk", "1mo"],
        index=1, 
        help="How the volume is bucketed (5-minute, daily, weekly, monthly).",
    )
    if range_mode == "Quick range":
        range_options = ["1d"] if interval == "5m" else ["5d", "1mo", "3mo", "6mo", "YTD", "1y", "2y", "5y", "max"]
        quick_choice = st.selectbox("Select range", range_options, index=0 if interval == "5m" else 1)
        start_date = None
        end_date = None
    else:
        today = dt.date.today()
        default_start = today - dt.timedelta(days=90)
        start_date = st.date_input("Start date", value=default_start, max_value=today)
        end_date = st.date_input("End date", value=today, max_value=today)
        quick_choice = None

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

    show_rising_line = st.checkbox(
        "Show rising avg line",
        value=False,
    )

    show_falling_line = st.checkbox(
        "Show falling avg line",
        value=False,
    )
    fetch_clicked = st.button("Get data", type="primary", use_container_width=True)

    st.divider()

    st.subheader("Volume % Change ")
    try:
        mdf = fetch_minute_volume(ticker_input) if ticker_input else pd.DataFrame()
    except Exception:
        mdf = pd.DataFrame()
    if not mdf.empty:
        trading_days = sorted(mdf["Day"].unique())
        selected_day = st.date_input(
            "Trading day",
            value=trading_days[-1],
            min_value=trading_days[0],
            max_value=trading_days[-1],
        )

    else:
        trading_days = []
        selected_day = None



render()

st.divider()
st.caption(
    "Data from Yahoo Finance via `yfinance`"
)
st.caption(
    "By Carson Lam"
)