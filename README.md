# Stock Volume Tracker

A tiny, free stock tool. Enter a ticker + a time period, get the trading
volume over that period (chart + summary stats + CSV download).

## What it uses (all free, no signup)
- **Data**: [Yahoo Finance](https://finance.yahoo.com) via the open-source
  `yfinance` Python library 
- **App framework**: [Streamlit](https://streamlit.io) 
- **Charting**: Plotly.

## How to run it

1. Make sure you have Python 3.9+ installed.
2. Open a terminal in this folder and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   streamlit run app.py
   ```

4. open automatically in your browser (usually at
   `http://localhost:8501`). If it doesn't, just paste that URL in manually.

## Using it

1. Type a ticker symbol in the sidebar (e.g. `AAPL`, `TSLA`, `MSFT`).
2. Pick a time period. Quick preset (1mo, 6mo, 1y, etc.) or
   custom start/end dates.
3. Choose a bar interval (daily / weekly / monthly buckets).
4. Click **Get data**.

You'll see:
- Total volume traded over the period
- Average volume per bar
- Highest/lowest volume day
- A bar chart of volume over time
- A raw data table you can download as CSV

## Notes / limitations
- Yahoo Finance data can occasionally be delayed or unavailable.
  if a ticker fails, try again in a moment or double-check the symbol.
- Everything runs locally on your machine. no data is sent anywhere
  except the request to Yahoo Finance for the price/volume history.
