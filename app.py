
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="PRO TRADER AI SYSTEM",
    page_icon="📈",
    layout="wide"
)

st.title("🚀 PRO TRADER AI SYSTEM")
st.caption("NSE + BSE | 100+ Technical Metrics | EOD Scanner")

# ---------------- CONFIGURATION ----------------

DEFAULT_STOCKS = """
RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,
SBIN,AXISBANK,KOTAKBANK,LT,BEL,
HAL,BDL,CDSL,NSDL,CAMS,INOXINDIA,
WELCORP,SANsera,GABRIEL,ASAHIINDIA,
PPLPHARMA,COFORGE,PERSISTENT,
TRENT,DIXON,ABB,SIEMENS,POWERGRID,
NTPC,TATAPOWER,JSWSTEEL,TATASTEEL
"""

st.sidebar.header("⚙️ Scanner Settings")

exchange = st.sidebar.selectbox(
    "Select Exchange",
    ["NSE", "BSE"]
)

period = st.sidebar.selectbox(
    "Historical Data",
    ["6mo", "1y", "2y", "5y"],
    index=1
)

stocks_input = st.sidebar.text_area(
    "Enter Stock Symbols (comma separated)",
    DEFAULT_STOCKS,
    height=180
)

min_score = st.sidebar.slider(
    "Minimum Technical Score",
    0, 100, 60
)

# ---------------- INDICATOR ENGINE ----------------

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # Moving averages
    for p in [5, 8, 9, 10, 12, 13, 15, 20, 21, 25,
              30, 34, 40, 44, 50, 55, 60, 75, 89,
              100, 120, 150, 180, 200]:

        df[f"SMA_{p}"] = close.rolling(p).mean()
        df[f"EMA_{p}"] = close.ewm(span=p, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for p in [5, 7, 9, 11, 14, 16, 18, 21, 25, 30]:
        avg_gain = gain.ewm(alpha=1/p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df[f"RSI_{p}"] = 100 - (100 / (1 + rs))

    # MACD variations
    for fast, slow in [(8,21), (12,26), (5,35), (19,39)]:
        macd = (
            close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean()
        )
        signal = macd.ewm(span=9, adjust=False).mean()
        df[f"MACD_{fast}_{slow}"] = macd
        df[f"MACD_SIGNAL_{fast}_{slow}"] = signal
        df[f"MACD_HIST_{fast}_{slow}"] = macd - signal

    # Bollinger Bands
    for p in [10, 20, 30]:
        mid = close.rolling(p).mean()
        std = close.rolling(p).std()
        df[f"BB_UPPER_{p}"] = mid + 2 * std
        df[f"BB_LOWER_{p}"] = mid - 2 * std
        df[f"BB_WIDTH_{p}"] = (
            (df[f"BB_UPPER_{p}"] - df[f"BB_LOWER_{p}"])
            / mid.replace(0, np.nan)
        ) * 100

    # Stochastic
    for p in [5, 9, 14, 21]:
        lowest = low.rolling(p).min()
        highest = high.rolling(p).max()
        df[f"STOCH_{p}"] = (
            (close - lowest) /
            (highest - lowest).replace(0, np.nan)
        ) * 100

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    for p in [7, 10, 14, 20]:
        df[f"ATR_{p}"] = tr.rolling(p).mean()

    # ADX / Directional indicators
    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    for p in [7, 14, 20]:
        atr = tr.rolling(p).mean()
        plus_di = 100 * plus_dm.rolling(p).sum() / atr.rolling(p).sum()
        minus_di = 100 * minus_dm.rolling(p).sum() / atr.rolling(p).sum()

        dx = (
            (plus_di - minus_di).abs()
            / (plus_di + minus_di).replace(0, np.nan)
        ) * 100

        df[f"PLUS_DI_{p}"] = plus_di
        df[f"MINUS_DI_{p}"] = minus_di
        df[f"ADX_{p}"] = dx.rolling(p).mean()

    # ROC / Momentum
    for p in [5, 10, 12, 15, 20, 30, 50, 100]:
        df[f"ROC_{p}"] = close.pct_change(p) * 100
        df[f"MOM_{p}"] = close - close.shift(p)

    # CCI
    tp = (high + low + close) / 3
    for p in [14, 20]:
        mean = tp.rolling(p).mean()
        mad = tp.rolling(p).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))),
            raw=True
        )
        df[f"CCI_{p}"] = (tp - mean) / (0.015 * mad)

    # Williams %R
    for p in [7, 14, 21]:
        hh = high.rolling(p).max()
        ll = low.rolling(p).min()
        df[f"WILLIAMS_{p}"] = (
            (hh - close) / (hh - ll).replace(0, np.nan)
        ) * -100

    # Money Flow Index
    typical = (high + low + close) / 3
    money_flow = typical * volume
    positive = money_flow.where(typical > typical.shift(), 0)
    negative = money_flow.where(typical < typical.shift(), 0)

    for p in [10, 14, 20]:
        ratio = (
            positive.rolling(p).sum()
            / negative.rolling(p).sum().replace(0, np.nan)
        )
        df[f"MFI_{p}"] = 100 - (100 / (1 + ratio))

    # Volume indicators
    df["OBV"] = (
        np.sign(close.diff()).fillna(0) * volume
    ).cumsum()

    for p in [5, 10, 20, 50]:
        df[f"VOLUME_SMA_{p}"] = volume.rolling(p).mean()

    df["VOLUME_RATIO"] = (
        volume / volume.rolling(20).mean()
    )

    # VWAP
    df["VWAP"] = (
        (typical * volume).rolling(20).sum()
        / volume.rolling(20).sum()
    )

    # Ichimoku
    df["ICHIMOKU_TENKAN"] = (
        high.rolling(9).max() + low.rolling(9).min()
    ) / 2

    df["ICHIMOKU_KIJUN"] = (
        high.rolling(26).max() + low.rolling(26).min()
    ) / 2

    df["ICHIMOKU_SPAN_A"] = (
        df["ICHIMOKU_TENKAN"] + df["ICHIMOKU_KIJUN"]
    ) / 2

    df["ICHIMOKU_SPAN_B"] = (
        high.rolling(52).max() + low.rolling(52).min()
    ) / 2

    # Price volatility
    for p in [5, 10, 20, 30, 60]:
        df[f"VOLATILITY_{p}"] = (
            close.pct_change().rolling(p).std() * np.sqrt(252) * 100
        )

    return df

# ---------------- SCORING ENGINE ----------------

def analyze_stock(symbol):

    suffix = ".NS" if exchange == "NSE" else ".BO"
    ticker = symbol.strip().upper() + suffix

    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if data.empty or len(data) < 60:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = calculate_indicators(data)
        last = df.iloc[-1]
        prev = df.iloc[-2]

        score = 0
        signals = []

        # Trend signals
        if last["Close"] > last["EMA_20"]:
            score += 5
            signals.append("Above EMA20")

        if last["Close"] > last["EMA_50"]:
            score += 5
            signals.append("Above EMA50")

        if last["Close"] > last["SMA_200"]:
            score += 10
            signals.append("Above SMA200")

        if last["EMA_20"] > last["EMA_50"]:
            score += 5
            signals.append("EMA Bullish")

        if last["EMA_50"] > last["SMA_200"]:
            score += 5
            signals.append("Golden Trend")

        # RSI
        if 50 < last["RSI_14"] < 70:
            score += 10
            signals.append("RSI Bullish")

        elif last["RSI_14"] <= 30:
            score += 5
            signals.append("RSI Oversold")

        # MACD
        if last["MACD_12_26"] > last["MACD_SIGNAL_12_26"]:
            score += 10
            signals.append("MACD Bullish")

        # ADX
        if last["ADX_14"] > 25:
            score += 10
            signals.append("Strong Trend")

        # Volume breakout
        if last["VOLUME_RATIO"] > 1.5:
            score += 10
            signals.append("Volume Breakout")

        # Bollinger breakout
        if last["Close"] > last["BB_UPPER_20"]:
            score += 5
            signals.append("BB Breakout")

        # Stochastic
        if last["STOCH_14"] > 50:
            score += 5
            signals.append("Stochastic Bullish")

        # Money flow
        if last["MFI_14"] > 50:
            score += 5
            signals.append("Money Flow Positive")

        # Price momentum
        if last["ROC_20"] > 0:
            score += 5
            signals.append("Positive Momentum")

        # VWAP
        if last["Close"] > last["VWAP"]:
            score += 5
            signals.append("Above VWAP")

        # Breakout of recent 20-day high (excluding today)
        previous_high = df["High"].shift(1).rolling(20).max().iloc[-1]

        if last["Close"] > previous_high:
            score += 10
            signals.append("20 Day Breakout")

        score = min(score, 100)

        if score >= 75:
            recommendation = "STRONG BULLISH"
        elif score >= 60:
            recommendation = "BULLISH"
        elif score >= 40:
            recommendation = "NEUTRAL"
        else:
            recommendation = "WEAK"

        return {
            "Symbol": symbol,
            "Price": round(float(last["Close"]), 2),
            "Change %": round(
                float((last["Close"] / prev["Close"] - 1) * 100), 2
            ),
            "RSI": round(float(last["RSI_14"]), 2),
            "ADX": round(float(last["ADX_14"]), 2),
            "Volume Ratio": round(float(last["VOLUME_RATIO"]), 2),
            "Technical Score": score,
            "Signal": recommendation,
            "Indicators": ", ".join(signals)
        }

    except Exception:
        return None

# ---------------- DASHBOARD ----------------

st.divider()

c1, c2, c3 = st.columns(3)

c1.metric("Exchange", exchange)
c2.metric("Technical Indicators", "100+")
c3.metric("Data Type", "Daily EOD")

if st.button("🚀 START PRO SCANNER", type="primary", use_container_width=True):

    stocks = [
        s.strip().upper()
        for s in stocks_input.replace("\n", ",").split(",")
        if s.strip()
    ]

    stocks = list(dict.fromkeys(stocks))

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(stocks):

        status.write(f"Scanning {symbol} ({i+1}/{len(stocks)})")

        result = analyze_stock(symbol)

        if result:
            results.append(result)

        progress.progress((i + 1) / len(stocks))

    status.success("Scanner completed!")

    if results:

        result_df = pd.DataFrame(results)

        result_df = result_df.sort_values(
            "Technical Score", ascending=False
        )

        st.session_state["scanner_results"] = result_df

    else:
        st.warning("No stock data received. Please try again.")

if "scanner_results" in st.session_state:

    result_df = st.session_state["scanner_results"]

    st.subheader("📊 PRO TRADER Results")

    st.metric("Stocks Scanned", len(result_df))

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🔥 Top Breakout Opportunities")

    top = result_df[
        result_df["Technical Score"] >= min_score
    ]

    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "📥 Download Scanner Report CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        file_name="PRO_TRADER_SCANNER.csv",
        mime="text/csv"
    )

    st.caption(
        "Technical screening only. Signals are not guaranteed returns "
        "or personalized investment advice."
    )

st.divider()
st.caption(
    "PRO TRADER AI SYSTEM | Built for NSE & BSE EOD Analysis"
)
