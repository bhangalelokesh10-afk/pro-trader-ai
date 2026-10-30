
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="PRO TRADER AI SYSTEM",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 PRO TRADER AI SYSTEM")
st.caption("ADVANCED VERSION 2.0 | NSE + BSE | EOD ANALYSIS")

# =====================================================
# CONFIGURATION
# =====================================================

DEFAULT_STOCKS = """
RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,AXISBANK,
KOTAKBANK,LT,BEL,HAL,BDL,CDSL,NSDL,CAMS,INOXINDIA,
WELCORP,SANSERA,GABRIEL,ASAHIINDIA,PPLPHARMA,
COFORGE,PERSISTENT,TRENT,DIXON,ABB,SIEMENS,
POWERGRID,NTPC,TATAPOWER,JSWSTEEL,TATASTEEL
"""

st.sidebar.header("⚙️ Scanner Configuration")

exchange = st.sidebar.selectbox(
    "Select Exchange",
    ["NSE", "BSE"]
)

period = st.sidebar.selectbox(
    "Historical Data",
    ["1y", "2y", "5y"],
    index=1
)

stocks_input = st.sidebar.text_area(
    "Stock Symbols",
    DEFAULT_STOCKS,
    height=180
)

min_score = st.sidebar.slider(
    "Minimum Technical Score",
    0, 100, 60
)

max_stocks = st.sidebar.selectbox(
    "Display Top Stocks",
    [10, 25, 50, 100, 500],
    index=3
)

st.sidebar.info(
    "Use Yahoo Finance compatible symbols. "
    "BSE symbols may require their exact BSE ticker."
)

# =====================================================
# TECHNICAL INDICATOR ENGINE
# =====================================================

def calculate_indicators(df):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # SMA + EMA
    periods = [
        5, 8, 9, 10, 12, 13, 15, 20, 21,
        25, 30, 34, 40, 44, 50, 55, 60,
        75, 89, 100, 120, 150, 180, 200
    ]

    for p in periods:
        df[f"SMA_{p}"] = close.rolling(p).mean()
        df[f"EMA_{p}"] = close.ewm(
            span=p, adjust=False
        ).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    for p in [5, 7, 9, 11, 14, 16, 18, 21, 25, 30]:

        avg_gain = gain.ewm(
            alpha=1/p, adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/p, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        df[f"RSI_{p}"] = 100 - (100 / (1 + rs))

    # MACD
    for fast, slow in [
        (8, 21), (12, 26), (5, 35), (19, 39)
    ]:

        macd = (
            close.ewm(span=fast, adjust=False).mean()
            - close.ewm(span=slow, adjust=False).mean()
        )

        signal = macd.ewm(
            span=9, adjust=False
        ).mean()

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
            (close - lowest)
            / (highest - lowest).replace(0, np.nan)
        ) * 100

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    for p in [7, 10, 14, 20]:
        df[f"ATR_{p}"] = tr.rolling(p).mean()

    # ADX / Directional Indicators
    up = high.diff()
    down = -low.diff()

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    for p in [7, 14, 20]:

        atr = tr.rolling(p).mean()

        plus_di = (
            100 * plus_dm.rolling(p).sum()
            / atr.rolling(p).sum().replace(0, np.nan)
        )

        minus_di = (
            100 * minus_dm.rolling(p).sum()
            / atr.rolling(p).sum().replace(0, np.nan)
        )

        dx = (
            (plus_di - minus_di).abs()
            / (plus_di + minus_di).replace(0, np.nan)
        ) * 100

        df[f"PLUS_DI_{p}"] = plus_di
        df[f"MINUS_DI_{p}"] = minus_di
        df[f"ADX_{p}"] = dx.rolling(p).mean()

    # ROC + Momentum
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

        df[f"CCI_{p}"] = (
            (tp - mean) / (0.015 * mad)
        )

    # Williams %R
    for p in [7, 14, 21]:

        hh = high.rolling(p).max()
        ll = low.rolling(p).min()

        df[f"WILLIAMS_{p}"] = (
            (hh - close)
            / (hh - ll).replace(0, np.nan)
        ) * -100

    # MFI
    typical = (high + low + close) / 3
    money_flow = typical * volume

    positive = money_flow.where(
        typical > typical.shift(), 0
    )

    negative = money_flow.where(
        typical < typical.shift(), 0
    )

    for p in [10, 14, 20]:

        ratio = (
            positive.rolling(p).sum()
            / negative.rolling(p).sum().replace(0, np.nan)
        )

        df[f"MFI_{p}"] = 100 - (100 / (1 + ratio))

    # OBV
    df["OBV"] = (
        np.sign(close.diff()).fillna(0) * volume
    ).cumsum()

    for p in [5, 10, 20, 50]:
        df[f"VOLUME_SMA_{p}"] = volume.rolling(p).mean()

    df["VOLUME_RATIO"] = (
        volume / volume.rolling(20).mean().replace(0, np.nan)
    )

    # VWAP
    df["VWAP"] = (
        (typical * volume).rolling(20).sum()
        / volume.rolling(20).sum().replace(0, np.nan)
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

    # Volatility
    for p in [5, 10, 20, 30, 60]:

        df[f"VOLATILITY_{p}"] = (
            close.pct_change().rolling(p).std()
            * np.sqrt(252) * 100
        )

    return df

# =====================================================
# FUNDAMENTAL ANALYSIS
# =====================================================

def fundamental_analysis(ticker):

    try:

        info = yf.Ticker(ticker).info or {}

        fields = {
            "ROE": info.get("returnOnEquity"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Earnings Growth": info.get("earningsGrowth"),
            "Profit Margin": info.get("profitMargins"),
            "Debt Equity": info.get("debtToEquity"),
            "Operating Cash Flow": info.get("operatingCashflow"),
            "PE Ratio": info.get("trailingPE"),
            "Market Cap": info.get("marketCap")
        }

        score = 0
        available_weight = 0

        rules = [
            ("ROE", 20, lambda x: x > 0.15),
            ("Revenue Growth", 20, lambda x: x > 0.10),
            ("Earnings Growth", 20, lambda x: x > 0.10),
            ("Profit Margin", 15, lambda x: x > 0.10),
            ("Debt Equity", 15, lambda x: x < 50),
            ("Operating Cash Flow", 10, lambda x: x > 0)
        ]

        for field, weight, condition in rules:

            value = fields[field]

            if value is not None and np.isfinite(value):

                available_weight += weight

                if condition(value):
                    score += weight

        if available_weight > 0:

            final_score = round(
                score / available_weight * 100, 2
            )

        else:
            final_score = np.nan

        return fields, final_score

    except Exception:

        return {}, np.nan

# =====================================================
# INSTITUTIONAL DATA IMPORT
# =====================================================

st.sidebar.divider()

st.sidebar.subheader("🏦 Institutional Data")

institution_file = st.sidebar.file_uploader(
    "Upload FII / DII / Promoter CSV",
    type=["csv"]
)

institution_data = None

if institution_file is not None:

    try:

        institution_data = pd.read_csv(institution_file)

        institution_data.columns = (
            institution_data.columns.str.strip()
        )

        required = [
            "Symbol",
            "FII Change %",
            "DII Change %",
            "Promoter Holding Change %"
        ]

        if all(c in institution_data.columns for c in required):

            institution_data["Symbol"] = (
                institution_data["Symbol"].astype(str).str.upper().str.strip()
            )

            institution_data = institution_data.drop_duplicates(
                "Symbol", keep="last"
            )

            st.sidebar.success("Institutional CSV Loaded")

        else:

            institution_data = None

            st.sidebar.error(
                "Required columns: Symbol, FII Change %, "
                "DII Change %, Promoter Holding Change %"
            )

    except Exception as e:

        st.sidebar.error(f"CSV Error: {e}")

# =====================================================
# PRICE DATA DOWNLOAD
# =====================================================

@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker, history_period):

    data = yf.download(
        ticker,
        period=history_period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):

        if ticker in data.columns.get_level_values(-1):

            data = data.xs(
                ticker, axis=1, level=-1
            )

        else:

            data.columns = data.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]

    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    return data.dropna(subset=["Close"])

# =====================================================
# STOCK ANALYSIS ENGINE
# =====================================================

def analyze_stock(symbol):

    suffix = ".NS" if exchange == "NSE" else ".BO"

    clean_symbol = symbol.strip().upper()

    if clean_symbol.endswith((".NS", ".BO")):

        ticker = clean_symbol

    else:

        ticker = clean_symbol + suffix

    try:

        data = download_stock_data(ticker, period)

        if data.empty or len(data) < 260:
            return None, f"{symbol}: Insufficient historical data"

        df = calculate_indicators(data)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close = float(last["Close"])

        if not np.isfinite(close) or close <= 0:
            return None, f"{symbol}: Invalid price"

        score = 0
        signals = []

        # TREND SCORE - 25
        if close > last["EMA_20"]:
            score += 5
            signals.append("Above EMA20")

        if close > last["EMA_50"]:
            score += 5
            signals.append("Above EMA50")

        if close > last["SMA_200"]:
            score += 10
            signals.append("Above SMA200")

        if last["EMA_20"] > last["EMA_50"]:
            score += 5
            signals.append("EMA Bullish")

        # RSI - 10
        if 50 < last["RSI_14"] < 70:
            score += 10
            signals.append("RSI Bullish")

        elif 40 <= last["RSI_14"] <= 50:
            score += 5

        # MACD - 10
        if last["MACD_12_26"] > last["MACD_SIGNAL_12_26"]:
            score += 10
            signals.append("MACD Bullish")

        # ADX - 10
        if last["ADX_14"] > 25:
            score += 10
            signals.append("Strong Trend")

        # VOLUME - 10
        if last["VOLUME_RATIO"] > 1.5:
            score += 10
            signals.append("Volume Breakout")

        # BOLLINGER - 5
        if close > last["BB_UPPER_20"]:
            score += 5
            signals.append("BB Breakout")

        # STOCHASTIC - 5
        if last["STOCH_14"] > 50:
            score += 5
            signals.append("Stochastic Bullish")

        # MFI - 5
        if last["MFI_14"] > 50:
            score += 5
            signals.append("Money Flow Positive")

        # MOMENTUM - 5
        if last["ROC_20"] > 0:
            score += 5
            signals.append("Positive Momentum")

        # VWAP - 5
        if close > last["VWAP"]:
            score += 5
            signals.append("Above VWAP")

        # BREAKOUT - 10
        high20 = df["High"].shift(1).rolling(20).max().iloc[-1]
        high50 = df["High"].shift(1).rolling(50).max().iloc[-1]
        high252 = df["High"].shift(1).rolling(252).max().iloc[-1]

        breakout20 = close > high20
        breakout50 = close > high50
        breakout52 = close > high252

        if breakout20:
            score += 4
            signals.append("20D Breakout")

        if breakout50:
            score += 3
            signals.append("50D Breakout")

        if breakout52:
            score += 3
            signals.append("52W Breakout")

        score = min(score, 100)

        # FUNDAMENTAL SCORE
        fundamentals, fundamental_score = fundamental_analysis(ticker)

        if np.isfinite(fundamental_score):

            overall_score = round(
                score * 0.70 + fundamental_score * 0.30, 2
            )

        else:

            overall_score = np.nan

        # INSTITUTIONAL DATA
        fii_change = np.nan
        dii_change = np.nan
        promoter_change = np.nan

        if institution_data is not None:

            match = institution_data[
                institution_data["Symbol"] == clean_symbol
            ]

            if not match.empty:

                row = match.iloc[0]

                fii_change = row["FII Change %"]
                dii_change = row["DII Change %"]
                promoter_change = row["Promoter Holding Change %"]

        if np.isfinite(promoter_change) and promoter_change < 0:

            institutional_signal = "Promoter Holding Reduced"

        elif (
            np.isfinite(fii_change) and fii_change > 0
        ) or (
            np.isfinite(dii_change) and dii_change > 0
        ):

            institutional_signal = "Institutional Buying Data Positive"

        else:

            institutional_signal = "No Confirmed Signal"

        # RECOMMENDATION
        if score >= 75:
            recommendation = "STRONG BULLISH"

        elif score >= 60:
            recommendation = "BULLISH"

        elif score >= 40:
            recommendation = "NEUTRAL"

        else:
            recommendation = "WEAK"

        # RESULT
        result = {

            "Symbol": clean_symbol,
            "Price": round(close, 2),

            "Change %": round(
                (close / float(prev["Close"]) - 1) * 100, 2
            ),

            "Technical Score": score,
            "Fundamental Score": fundamental_score,
            "Overall Score": overall_score,

            "RSI": round(float(last["RSI_14"]), 2),
            "ADX": round(float(last["ADX_14"]), 2),

            "Volume Ratio": round(
                float(last["VOLUME_RATIO"]), 2
            ),

            "Breakout 20D": breakout20,
            "Breakout 50D": breakout50,
            "Breakout 52W": breakout52,

            "FII Change %": fii_change,
            "DII Change %": dii_change,
            "Promoter Change %": promoter_change,

            "Institutional Signal": institutional_signal,
            "Signal": recommendation,

            "Market Cap": fundamentals.get("Market Cap"),
            "ROE": fundamentals.get("ROE"),
            "Revenue Growth": fundamentals.get("Revenue Growth"),
            "Earnings Growth": fundamentals.get("Earnings Growth"),
            "PE Ratio": fundamentals.get("PE Ratio"),
            "Debt Equity": fundamentals.get("Debt Equity"),

            "Indicators": ", ".join(signals)
        }

        return result, None

    except Exception as e:

        return None, f"{symbol}: {str(e)[:150]}"

# =====================================================
# DASHBOARD
# =====================================================

st.divider()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Exchange", exchange)
c2.metric("Technical Metrics", "140+")
c3.metric("Data Type", "Daily EOD")
c4.metric("History", period)

st.info(
    "Scanner uses historical daily data. "
    "Latest available market session is analyzed."
)

# =====================================================
# START SCANNER
# =====================================================

if st.button(
    "🚀 START PRO TRADER SCANNER",
    type="primary",
    use_container_width=True
):

    stocks = [
        s.strip().upper()
        for s in stocks_input.replace("\n", ",").split(",")
        if s.strip()
    ]

    stocks = list(dict.fromkeys(stocks))

    if not stocks:

        st.error("Please enter stock symbols.")

    else:

        results = []
        errors = []

        progress = st.progress(0)
        status = st.empty()

        for i, symbol in enumerate(stocks):

            status.write(
                f"Scanning {symbol} ({i+1}/{len(stocks)})"
            )

            result, error = analyze_stock(symbol)

            if result is not None:
                results.append(result)

            if error:
                errors.append(error)

            progress.progress((i + 1) / len(stocks))

        status.success("PRO Scanner Completed!")

        if results:

            result_df = pd.DataFrame(results)

            result_df = result_df.sort_values(
                "Technical Score",
                ascending=False
            ).reset_index(drop=True)

            st.session_state["scanner_results"] = result_df
            st.session_state["scanner_errors"] = errors

        else:

            st.error(
                "No stock data received. Check symbols or data connection."
            )

# =====================================================
# DISPLAY RESULTS
# =====================================================

if "scanner_results" in st.session_state:

    result_df = st.session_state["scanner_results"]

    st.divider()

    st.header("📊 PRO TRADER SCANNER RESULTS")

    a, b, c, d = st.columns(4)

    a.metric("Stocks Scanned", len(result_df))

    b.metric(
        "Bullish Stocks",
        int((result_df["Technical Score"] >= 60).sum())
    )

    c.metric(
        "Strong Bullish",
        int((result_df["Technical Score"] >= 75).sum())
    )

    d.metric(
        "Breakout Stocks",
        int(result_df["Breakout 20D"].sum())
    )

    # TOP BREAKOUT
    st.subheader("🔥 Advanced Breakout Opportunities")

    breakout_df = result_df[
        result_df["Breakout 20D"]
        | result_df["Breakout 50D"]
        | result_df["Breakout 52W"]
    ].copy()

    if not breakout_df.empty:

        st.dataframe(
            breakout_df.head(max_stocks),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning("No confirmed breakout in scanned stocks.")

    # TOP TECHNICAL
    st.subheader("🏆 Top Technical Opportunities")

    top_df = result_df[
        result_df["Technical Score"] >= min_score
    ].head(max_stocks)

    st.dataframe(
        top_df,
        use_container_width=True,
        hide_index=True
    )

    # FUNDAMENTAL
    st.subheader("💎 Fundamental Analysis")

    fundamental_df = result_df.dropna(
        subset=["Fundamental Score"]
    ).sort_values(
        "Fundamental Score",
        ascending=False
    )

    if not fundamental_df.empty:

        st.dataframe(
            fundamental_df.head(max_stocks),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Fundamental data unavailable for these stocks."
        )

    # INSTITUTIONAL
    st.subheader("🏦 Institutional & Promoter Analysis")

    st.dataframe(
        result_df[
            [
                "Symbol",
                "FII Change %",
                "DII Change %",
                "Promoter Change %",
                "Institutional Signal"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # CHART
    st.subheader("📈 Technical Score Ranking")

    chart_data = result_df.set_index("Symbol")[
        "Technical Score"
    ]

    st.bar_chart(chart_data.head(20))

    # DOWNLOAD
    st.subheader("📥 Download Reports")

    csv = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Full Scanner CSV",
        csv,
        file_name="PRO_TRADER_AI_REPORT.csv",
        mime="text/csv",
        use_container_width=True
    )

    # ERRORS
    errors = st.session_state.get("scanner_errors", [])

    if errors:

        with st.expander("⚠️ Stocks with Data Errors"):

            for error in errors:
                st.write(error)

    st.caption(
        "Technical and fundamental screening only. "
        "Scores are research filters, not guaranteed returns "
        "or personalized investment advice."
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "PRO TRADER AI SYSTEM | ADVANCED VERSION 2.0 | "
    "NSE + BSE | EOD MARKET ANALYSIS"
)
