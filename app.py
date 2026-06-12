"""
app.py — NIFTY-50 Investment Intelligence Dashboard
Phase 6 of the Data-Driven Investment Intelligence project.

Run:
    streamlit run app.py

Expects these CSV files in the same folder (or configure DATA_DIR below):
    nifty50_processed_data.csv
    nifty50_signals.csv
    nifty50_model_results.csv
    nifty50_portfolios.csv
    nifty50_portfolio_performance.csv
    nifty50_stock_risk.csv
    nifty50_portfolio_risk.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))   # same folder as app.py

st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── PALETTE & THEME ───────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0D1117",
    "card":        "#161B22",
    "border":      "#30363D",
    "accent":      "#58A6FF",
    "green":       "#3FB950",
    "red":         "#F85149",
    "yellow":      "#D29922",
    "text":        "#E6EDF3",
    "muted":       "#8B949E",
    "conservative":"#3FB950",
    "balanced":    "#D29922",
    "aggressive":  "#F85149",
}

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── base ── */
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {COLORS['bg']};
      color: {COLORS['text']};
      font-family: 'Inter', 'Segoe UI', sans-serif;
  }}
  [data-testid="stSidebar"] {{
      background-color: {COLORS['card']};
      border-right: 1px solid {COLORS['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {COLORS['text']}; }}

  /* ── metric cards ── */
  .kpi-card {{
      background: {COLORS['card']};
      border: 1px solid {COLORS['border']};
      border-radius: 10px;
      padding: 18px 20px 14px;
      margin-bottom: 12px;
  }}
  .kpi-label {{
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {COLORS['muted']};
      margin-bottom: 4px;
  }}
  .kpi-value {{
      font-size: 26px;
      font-weight: 700;
      color: {COLORS['text']};
      line-height: 1.1;
  }}
  .kpi-sub {{
      font-size: 12px;
      color: {COLORS['muted']};
      margin-top: 4px;
  }}
  .positive {{ color: {COLORS['green']}; }}
  .negative {{ color: {COLORS['red']}; }}

  /* ── section headers ── */
  .section-header {{
      font-size: 18px;
      font-weight: 700;
      color: {COLORS['text']};
      border-left: 3px solid {COLORS['accent']};
      padding-left: 10px;
      margin: 24px 0 14px;
  }}

  /* ── signal badges ── */
  .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
  }}
  .badge-buy  {{ background:#1a3a1e; color:{COLORS['green']}; border:1px solid {COLORS['green']}; }}
  .badge-hold {{ background:#3a3010; color:{COLORS['yellow']};border:1px solid {COLORS['yellow']};}}
  .badge-sell {{ background:#3a1010; color:{COLORS['red']};   border:1px solid {COLORS['red']};   }}

  /* ── table ── */
  .stDataFrame {{ background: {COLORS['card']} !important; }}
  thead tr th {{
      background: {COLORS['border']} !important;
      color: {COLORS['text']} !important;
  }}

  /* ── hide streamlit chrome ── */
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.5rem; }}
</style>
""", unsafe_allow_html=True)

# ── SECTOR MAP ────────────────────────────────────────────────────────────────
SECTOR_MAP = {
    'ADANIPORTS':'Infrastructure','ASIANPAINT':'Consumer Goods','AXISBANK':'Banking',
    'BAJAJ-AUTO':'Automobile','BAJAJFINSV':'Financial Services','BAJFINANCE':'Financial Services',
    'BHARTIARTL':'Telecom','BPCL':'Energy','BRITANNIA':'Consumer Goods','CIPLA':'Pharma',
    'COALINDIA':'Energy','DRREDDY':'Pharma','EICHERMOT':'Automobile','GAIL':'Energy',
    'GRASIM':'Cement','HCLTECH':'IT','HDFC':'Financial Services','HDFCBANK':'Banking',
    'HDFCLIFE':'Insurance','HEROMOTOCO':'Automobile','HINDALCO':'Metals',
    'HINDUNILVR':'Consumer Goods','ICICIBANK':'Banking','INDUSINDBK':'Banking','INFY':'IT',
    'IOC':'Energy','ITC':'Consumer Goods','JSWSTEEL':'Metals','KOTAKBANK':'Banking',
    'LT':'Infrastructure','M&M':'Automobile','MARUTI':'Automobile','NESTLEIND':'Consumer Goods',
    'NTPC':'Energy','ONGC':'Energy','POWERGRID':'Energy','RELIANCE':'Energy',
    'SBILIFE':'Insurance','SBIN':'Banking','SHREECEM':'Cement','SUNPHARMA':'Pharma',
    'TATACONSUM':'Consumer Goods','TATAMOTORS':'Automobile','TATASTEEL':'Metals',
    'TCS':'IT','TECHM':'IT','TITAN':'Consumer Goods','ULTRACEMCO':'Cement',
    'UPL':'Chemicals','WIPRO':'IT','ZEEL':'Media',
}

PROFILE_COLORS = {
    "Conservative": COLORS["conservative"],
    "Balanced":     COLORS["balanced"],
    "Aggressive":   COLORS["aggressive"],
}

# ── PLOTLY TEMPLATE ───────────────────────────────────────────────────────────
def dark_fig(fig, height=400, margin=None):
    m = margin or dict(l=40, r=20, t=40, b=40)
    fig.update_layout(
        height=height,
        margin=m,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="Inter, Segoe UI, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"]),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"]),
    )
    return fig

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    def safe_read(name):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return pd.read_csv(path)
        return None

    price_raw   = safe_read("nifty50_processed_data.csv")
    signals     = safe_read("nifty50_signals.csv")
    model_res   = safe_read("nifty50_model_results.csv")
    portfolios  = safe_read("nifty50_portfolios.csv")
    port_perf   = safe_read("nifty50_portfolio_performance.csv")
    stock_risk  = safe_read("nifty50_stock_risk.csv")
    port_risk   = safe_read("nifty50_portfolio_risk.csv")

    if price_raw is not None:
        price_raw["Date"] = pd.to_datetime(price_raw["Date"])
        price_raw.sort_values(["Symbol", "Date"], inplace=True)
        price_raw.reset_index(drop=True, inplace=True)
        if "Sector" not in price_raw.columns:
            price_raw["Sector"] = price_raw["Symbol"].map(SECTOR_MAP)

    return price_raw, signals, model_res, portfolios, port_perf, stock_risk, port_risk

# ── DEMO DATA (runs even without CSVs) ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def make_demo():
    """Generate synthetic data so the dashboard is always navigable."""
    rng  = np.random.default_rng(42)
    symbols = list(SECTOR_MAP.keys())
    dates = pd.date_range("2015-01-01", "2021-04-30", freq="B")

    rows = []
    for sym in symbols:
        price = 100.0
        for d in dates:
            ret   = rng.normal(0.0004, 0.015)
            price = max(price * (1 + ret), 1)
            rows.append({"Symbol": sym, "Date": d, "Close": round(price, 2),
                         "Open": round(price * (1 + rng.normal(0, 0.005)), 2),
                         "High": round(price * (1 + abs(rng.normal(0, 0.008))), 2),
                         "Low":  round(price * (1 - abs(rng.normal(0, 0.008))), 2),
                         "Volume": int(rng.integers(100_000, 5_000_000)),
                         "Daily_Return": ret,
                         "Sector": SECTOR_MAP.get(sym, "Other")})
    df = pd.DataFrame(rows)

    # signals
    sig_rows = []
    for sym in symbols:
        r = rng.choice(["BUY", "HOLD", "SELL"], p=[0.4, 0.35, 0.25])
        sig_rows.append({"Symbol": sym, "Signal": r,
                         "RSI": round(rng.uniform(25, 75), 1),
                         "MACD": round(rng.uniform(-5, 5), 2),
                         "Momentum_20": round(rng.uniform(-0.1, 0.2), 3),
                         "Sector": SECTOR_MAP.get(sym, "Other")})
    signals = pd.DataFrame(sig_rows)

    # model results
    mr = pd.DataFrame({
        "Symbol": symbols,
        "RMSE": rng.uniform(5, 80, len(symbols)).round(2),
        "R2":   rng.uniform(0.75, 0.99, len(symbols)).round(4),
        "Directional_Acc": rng.uniform(50, 75, len(symbols)).round(2),
    })

    # portfolios
    port_rows = []
    for profile, syms, wts in [
        ("Conservative", ["HDFCBANK","TCS","HINDUNILVR","INFY","CIPLA","NESTLEIND","WIPRO"],
                         [0.20,0.18,0.15,0.14,0.13,0.11,0.09]),
        ("Balanced",     ["RELIANCE","HDFCBANK","TCS","INFY","ICICIBANK","KOTAKBANK","LT","BAJFINANCE"],
                         [0.18,0.15,0.14,0.12,0.11,0.10,0.10,0.10]),
        ("Aggressive",   ["TATASTEEL","ADANIPORTS","TATAMOTORS","HINDALCO","JSWSTEEL","BAJAJ-AUTO"],
                         [0.22,0.20,0.18,0.16,0.14,0.10]),
    ]:
        for s, w in zip(syms, wts):
            port_rows.append({"Profile": profile, "Symbol": s, "Weight": w,
                               "Weight_Pct": round(w*100, 2),
                               "Sector": SECTOR_MAP.get(s, "Other")})
    portfolios = pd.DataFrame(port_rows)

    port_perf = pd.DataFrame({
        "Profile":         ["Conservative","Balanced","Aggressive"],
        "Expected_Return": [0.12, 0.18, 0.26],
        "Volatility":      [0.14, 0.20, 0.30],
        "Sharpe_Ratio":    [0.86, 0.90, 0.95],
    })

    # stock risk
    sr_rows = []
    for sym in symbols:
        sr_rows.append({
            "Symbol":       sym,
            "Annual_Return": round(rng.uniform(-5, 40), 2),
            "Volatility":   round(rng.uniform(15, 50), 2),
            "Sharpe_Ratio": round(rng.uniform(0.1, 1.8), 4),
            "Sortino_Ratio":round(rng.uniform(0.1, 2.5), 4),
            "Max_Drawdown": round(rng.uniform(-70, -10), 2),
            "VaR_95":       round(rng.uniform(-4, -1), 2),
            "CVaR_95":      round(rng.uniform(-6, -2), 2),
        })
    stock_risk = pd.DataFrame(sr_rows).set_index("Symbol")

    port_risk = pd.DataFrame({
        "Annual_Return":  [12.1, 17.8, 25.5],
        "Volatility":     [14.2, 19.6, 29.8],
        "Sharpe_Ratio":   [0.86, 0.90, 0.95],
        "Sortino_Ratio":  [1.10, 1.25, 1.40],
        "Max_Drawdown":   [-22, -35, -52],
        "VaR_95":         [-1.5, -2.1, -3.2],
        "CVaR_95":        [-2.4, -3.1, -4.8],
    }, index=["Conservative","Balanced","Aggressive"])

    return df, signals, mr, portfolios, port_perf, stock_risk, port_risk

# ── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    price_raw, signals, model_res, portfolios, port_perf, stock_risk, port_risk = load_data()

# Fall back to demo if any file is missing
DEMO_MODE = any(x is None for x in [price_raw, signals, model_res, portfolios, port_perf, stock_risk, port_risk])
if DEMO_MODE:
    price_raw, signals, model_res, portfolios, port_perf, stock_risk, port_risk = make_demo()

# normalise stock_risk index
if "Symbol" in stock_risk.columns:
    stock_risk = stock_risk.set_index("Symbol")
if "Profile" in port_risk.columns:
    port_risk = port_risk.set_index("Profile")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 6px'>
      <div style='font-size:26px'>📈</div>
      <div style='font-size:15px;font-weight:700;letter-spacing:0.03em;
                  color:#E6EDF3'>NIFTY-50 Intelligence</div>
      <div style='font-size:11px;color:#8B949E;margin-top:2px'>
        Investment Analytics Platform
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📊 Stock Explorer",
         "🤖 Predictor & Signals",
         "💼 Portfolio Builder",
         "⚠️ Risk Dashboard"],
        label_visibility="collapsed",
    )

    st.divider()
    if DEMO_MODE:
        st.warning("⚠️ Demo mode — place your CSV files next to app.py to load real data.", icon="ℹ️")
    else:
        st.success("✅ Live data loaded", icon="✅")

    st.caption("NIFTY-50 · 2000 – 2021")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.markdown("# NIFTY-50 Investment Intelligence Platform")
    st.markdown(f"<span style='color:{COLORS['muted']};font-size:14px'>"
                "Historical analysis · Model predictions · Portfolio optimisation · Risk analytics"
                "</span>", unsafe_allow_html=True)
    st.divider()

    # ── KPI row
    symbols       = price_raw["Symbol"].nunique()
    date_min      = price_raw["Date"].min().strftime("%b %Y")
    date_max      = price_raw["Date"].max().strftime("%b %Y")
    total_records = len(price_raw)
    sectors       = price_raw["Sector"].nunique() if "Sector" in price_raw.columns else "—"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, sub in [
        (c1, "STOCKS COVERED",    f"{symbols}",           f"{date_min} → {date_max}"),
        (c2, "TRADING RECORDS",   f"{total_records:,}",   "Daily OHLCV rows"),
        (c3, "SECTORS",           f"{sectors}",           "Across NIFTY-50"),
        (c4, "MODELS EVALUATED",  f"{len(model_res) if model_res is not None else '—'}", "XGBoost per stock"),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{label}</div>
          <div class='kpi-value'>{val}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Sector composition
    st.markdown("<div class='section-header'>Sector Composition</div>", unsafe_allow_html=True)
    sector_counts = price_raw.drop_duplicates("Symbol")["Sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]

    fig_sec = px.bar(sector_counts, x="Sector", y="Count",
                     color="Count", color_continuous_scale="Blues",
                     labels={"Count": "# Stocks"})
    fig_sec.update_traces(marker_line_width=0)
    fig_sec.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
    st.plotly_chart(dark_fig(fig_sec, 360), use_container_width=True)

    # ── Return + Volatility overview
    st.markdown("<div class='section-header'>Total Return vs Annualised Volatility (all stocks)</div>",
                unsafe_allow_html=True)

    grp = price_raw.groupby("Symbol")
    total_ret  = grp["Close"].apply(lambda x: (x.iloc[-1]/x.iloc[0] - 1) * 100)
    ann_vol    = grp["Daily_Return"].std() * np.sqrt(252) * 100
    overview   = pd.DataFrame({"Total_Return": total_ret, "Volatility": ann_vol})
    overview["Sector"] = overview.index.map(SECTOR_MAP)
    overview.reset_index(inplace=True)

    fig_ov = px.scatter(overview, x="Volatility", y="Total_Return",
                        color="Sector", text="Symbol",
                        labels={"Total_Return": "Total Return (%)", "Volatility": "Ann. Volatility (%)"},
                        hover_data={"Symbol": True, "Sector": True})
    fig_ov.update_traces(textposition="top center", textfont_size=9, marker_size=9)
    fig_ov.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], line_width=1)
    st.plotly_chart(dark_fig(fig_ov, 480), use_container_width=True)

    # ── Correlation heatmap (last 5 years for speed)
    st.markdown("<div class='section-header'>Return Correlation Heatmap</div>",
                unsafe_allow_html=True)

    cutoff = price_raw["Date"].max() - pd.DateOffset(years=5)
    recent = price_raw[price_raw["Date"] >= cutoff]
    pivot  = recent.pivot_table(index="Date", columns="Symbol", values="Daily_Return")
    pivot  = pivot.dropna(thresh=int(0.7 * len(pivot)), axis=1)
    corr   = pivot.corr().round(2)

    fig_cor = px.imshow(corr, color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0, aspect="auto",
                        labels={"color": "Corr"})
    fig_cor.update_layout(coloraxis_colorbar=dict(thickness=12))
    st.plotly_chart(dark_fig(fig_cor, 520, margin=dict(l=10,r=10,t=40,b=10)),
                    use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — STOCK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Stock Explorer":

    st.markdown("# Stock Explorer")
    st.markdown(f"<span style='color:{COLORS['muted']};font-size:14px'>"
                "Deep-dive into any NIFTY-50 stock with price history and technical indicators."
                "</span>", unsafe_allow_html=True)
    st.divider()

    col_sym, col_per = st.columns([2, 1])
    with col_sym:
        all_syms = sorted(price_raw["Symbol"].unique())
        symbol   = st.selectbox("Select Stock", all_syms, index=all_syms.index("RELIANCE") if "RELIANCE" in all_syms else 0)
    with col_per:
        period = st.selectbox("Period", ["1Y", "3Y", "5Y", "Full"], index=2)

    df = price_raw[price_raw["Symbol"] == symbol].copy().sort_values("Date")
    cutoff_map = {"1Y": 365, "3Y": 365*3, "5Y": 365*5, "Full": 99999}
    cutoff = df["Date"].max() - pd.DateOffset(days=cutoff_map[period])
    df = df[df["Date"] >= cutoff]

    # KPIs
    latest     = df.iloc[-1]["Close"]
    start      = df.iloc[0]["Close"]
    chg_pct    = (latest / start - 1) * 100
    hi         = df["High"].max()
    lo         = df["Low"].min()
    avg_vol    = df["Volume"].mean() if "Volume" in df.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    sign = "positive" if chg_pct >= 0 else "negative"
    for col, lbl, val, sub in [
        (k1, "CURRENT PRICE",   f"₹{latest:,.2f}",     f"{period} change"),
        (k2, "PERIOD RETURN",   f"<span class='{sign}'>{chg_pct:+.1f}%</span>", f"₹{start:.2f} → ₹{latest:.2f}"),
        (k3, "PERIOD HIGH",     f"₹{hi:,.2f}",          "Intraday high"),
        (k4, "AVG DAILY VOLUME",f"{avg_vol/1e6:.1f}M",  "Shares traded"),
    ]:
        col.markdown(f"""
        <div class='kpi-card'>
          <div class='kpi-label'>{lbl}</div>
          <div class='kpi-value'>{val}</div>
          <div class='kpi-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    # Price + Moving Averages
    st.markdown("<div class='section-header'>Price & Moving Averages</div>",
                unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close",
                                   line=dict(color=COLORS["accent"], width=1.5)))
    for ma_col, ma_color, ma_name in [
        ("MA20",  "#F85149", "MA 20"),
        ("MA50",  "#D29922", "MA 50"),
        ("MA200", "#3FB950", "MA 200"),
    ]:
        if ma_col in df.columns:
            fig_price.add_trace(go.Scatter(x=df["Date"], y=df[ma_col],
                                           name=ma_name,
                                           line=dict(color=ma_color, width=1, dash="dot")))
    st.plotly_chart(dark_fig(fig_price, 380), use_container_width=True)

    # Bollinger Bands
    if all(c in df.columns for c in ["BB_Upper", "BB_Lower"]):
        st.markdown("<div class='section-header'>Bollinger Bands</div>",
                    unsafe_allow_html=True)
        fig_bb = go.Figure()
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"],
                                    name="Upper Band", line=dict(color="#8B949E", dash="dot", width=1)))
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"],
                                    name="Lower Band", line=dict(color="#8B949E", dash="dot", width=1),
                                    fill="tonexty", fillcolor="rgba(88,166,255,0.06)"))
        fig_bb.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                                    name="Close", line=dict(color=COLORS["accent"], width=1.5)))
        st.plotly_chart(dark_fig(fig_bb, 340), use_container_width=True)

    # RSI + MACD
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("<div class='section-header'>RSI (14)</div>",
                    unsafe_allow_html=True)
        if "RSI" in df.columns:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df["Date"], y=df["RSI"],
                                         name="RSI", line=dict(color=COLORS["yellow"], width=1.5)))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS["red"],   line_width=1)
            fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], line_width=1)
            fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", line_width=0)
            fig_rsi.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(dark_fig(fig_rsi, 300), use_container_width=True)

    with col_r:
        st.markdown("<div class='section-header'>MACD</div>",
                    unsafe_allow_html=True)
        if all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"]):
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram",
                                      marker_color=np.where(df["MACD_Hist"] >= 0,
                                                            COLORS["green"], COLORS["red"])))
            fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD"],
                                          name="MACD", line=dict(color=COLORS["accent"], width=1.5)))
            fig_macd.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"],
                                          name="Signal", line=dict(color=COLORS["yellow"], width=1)))
            st.plotly_chart(dark_fig(fig_macd, 300), use_container_width=True)

    # Volume
    if "Volume" in df.columns:
        st.markdown("<div class='section-header'>Volume</div>", unsafe_allow_html=True)
        fig_vol = go.Figure(go.Bar(x=df["Date"], y=df["Volume"],
                                   marker_color=COLORS["accent"], opacity=0.6, name="Volume"))
        st.plotly_chart(dark_fig(fig_vol, 250), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTOR & SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictor & Signals":

    st.markdown("# Predictor & Signals")
    st.markdown(f"<span style='color:{COLORS['muted']};font-size:14px'>"
                "XGBoost model performance across all NIFTY-50 stocks · BUY / HOLD / SELL signals"
                "</span>", unsafe_allow_html=True)
    st.divider()

    # ── Model Performance
    if model_res is not None and len(model_res):
        st.markdown("<div class='section-header'>Model Performance — All Stocks</div>",
                    unsafe_allow_html=True)

        avg_r2  = model_res["R2"].mean()
        avg_da  = model_res["Directional_Acc"].mean()
        best    = model_res.loc[model_res["R2"].idxmax(), "Symbol"]
        gt09    = (model_res["R2"] > 0.9).sum()

        m1, m2, m3, m4 = st.columns(4)
        for col, lbl, val, sub in [
            (m1, "AVG R² SCORE",       f"{avg_r2:.4f}",    f"{gt09} stocks > 0.90"),
            (m2, "AVG DIRECTIONAL ACC",f"{avg_da:.1f}%",   "Next-day direction"),
            (m3, "BEST MODEL",         best,               f"R²={model_res.loc[model_res['R2'].idxmax(),'R2']:.4f}"),
            (m4, "STOCKS MODELLED",    str(len(model_res)),"XGBoost regressor"),
        ]:
            col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value'>{val}</div>
              <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            fig_r2 = px.histogram(model_res, x="R2", nbins=25, color_discrete_sequence=[COLORS["accent"]],
                                   labels={"R2": "R² Score"})
            fig_r2.add_vline(x=avg_r2, line_dash="dash", line_color=COLORS["red"],
                             annotation_text=f"Mean={avg_r2:.3f}")
            fig_r2.update_traces(marker_line_width=0)
            st.plotly_chart(dark_fig(fig_r2, 320), use_container_width=True)

        with col_b:
            fig_da = px.histogram(model_res, x="Directional_Acc", nbins=25,
                                   color_discrete_sequence=[COLORS["green"]],
                                   labels={"Directional_Acc": "Directional Accuracy (%)"})
            fig_da.add_vline(x=avg_da, line_dash="dash", line_color=COLORS["red"],
                             annotation_text=f"Mean={avg_da:.1f}%")
            fig_da.update_traces(marker_line_width=0)
            st.plotly_chart(dark_fig(fig_da, 320), use_container_width=True)

        # Scatter R² vs Dir Acc
        mr_plot = model_res.copy()
        mr_plot["Sector"] = mr_plot["Symbol"].map(SECTOR_MAP)
        fig_sc = px.scatter(mr_plot, x="R2", y="Directional_Acc", color="Sector",
                             text="Symbol", size="RMSE",
                             labels={"R2": "R² Score", "Directional_Acc": "Directional Accuracy (%)"})
        fig_sc.update_traces(textposition="top center", textfont_size=8)
        st.plotly_chart(dark_fig(fig_sc, 420), use_container_width=True)

    # ── Signals
    if signals is not None and len(signals):
        st.markdown("<div class='section-header'>BUY / HOLD / SELL Signals</div>",
                    unsafe_allow_html=True)

        # filters
        f1, f2, f3 = st.columns([1, 1, 2])
        with f1:
            sig_filter = st.selectbox("Signal", ["All", "BUY", "HOLD", "SELL"])
        with f2:
            sec_opts = ["All"] + sorted(signals["Sector"].dropna().unique().tolist()) if "Sector" in signals.columns else ["All"]
            sec_filter = st.selectbox("Sector", sec_opts)

        sig_disp = signals.copy()
        if sig_filter != "All":
            sig_disp = sig_disp[sig_disp["Signal"] == sig_filter]
        if sec_filter != "All" and "Sector" in sig_disp.columns:
            sig_disp = sig_disp[sig_disp["Sector"] == sec_filter]

        # Signal distribution donut
        col_pie, col_tbl = st.columns([1, 2])
        with col_pie:
            sc = signals["Signal"].value_counts().reset_index()
            sc.columns = ["Signal", "Count"]
            color_map = {"BUY": COLORS["green"], "HOLD": COLORS["yellow"], "SELL": COLORS["red"]}
            sc["Color"] = sc["Signal"].map(color_map)
            fig_pie = go.Figure(go.Pie(labels=sc["Signal"], values=sc["Count"],
                                        marker_colors=sc["Color"],
                                        hole=0.55, textinfo="label+percent"))
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(dark_fig(fig_pie, 300), use_container_width=True)

        with col_tbl:
            def badge(sig):
                cls = {"BUY": "badge-buy", "HOLD": "badge-hold", "SELL": "badge-sell"}.get(sig, "")
                return f"<span class='badge {cls}'>{sig}</span>"

            show_cols = ["Symbol", "Signal"]
            if "Sector" in sig_disp.columns: show_cols.append("Sector")
            for c in ["RSI", "MACD", "Momentum_20"]:
                if c in sig_disp.columns: show_cols.append(c)

            tbl_html = sig_disp[show_cols].to_html(
                index=False, escape=False,
                formatters={"Signal": badge},
                classes="stDataFrame"
            )
            st.markdown(f"<div style='max-height:320px;overflow-y:auto'>{tbl_html}</div>",
                        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PORTFOLIO BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💼 Portfolio Builder":

    st.markdown("# Portfolio Builder")
    st.markdown(f"<span style='color:{COLORS['muted']};font-size:14px'>"
                "Mean-variance optimised portfolios for three investor profiles."
                "</span>", unsafe_allow_html=True)
    st.divider()

    # Profile comparison
    if port_perf is not None and len(port_perf):
        st.markdown("<div class='section-header'>Portfolio Comparison</div>",
                    unsafe_allow_html=True)

        cols = st.columns(3)
        for _, row in port_perf.iterrows():
            profile = row["Profile"]
            color   = PROFILE_COLORS.get(profile, COLORS["accent"])
            icon    = {"Conservative": "🟢", "Balanced": "🟡", "Aggressive": "🔴"}.get(profile, "")
            idx     = ["Conservative", "Balanced", "Aggressive"].index(profile) if profile in ["Conservative","Balanced","Aggressive"] else 0
            with cols[idx]:
                st.markdown(f"""
                <div class='kpi-card' style='border-color:{color}'>
                  <div class='kpi-label'>{icon} {profile.upper()}</div>
                  <div class='kpi-value' style='color:{color}'>{row["Expected_Return"]*100:.1f}%</div>
                  <div class='kpi-sub'>Expected annual return</div>
                  <hr style='border-color:{color}22;margin:10px 0'>
                  <div style='display:flex;justify-content:space-between;margin-top:6px'>
                    <span style='color:{COLORS["muted"]};font-size:12px'>Volatility</span>
                    <span style='font-size:12px'>{row["Volatility"]*100:.1f}%</span>
                  </div>
                  <div style='display:flex;justify-content:space-between;margin-top:4px'>
                    <span style='color:{COLORS["muted"]};font-size:12px'>Sharpe Ratio</span>
                    <span style='font-size:12px'>{row["Sharpe_Ratio"]:.3f}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

        # Bar comparison
        fig_comp = go.Figure()
        metrics_show = ["Expected_Return", "Volatility"]
        for profile, color in PROFILE_COLORS.items():
            row = port_perf[port_perf["Profile"] == profile]
            if len(row):
                fig_comp.add_trace(go.Bar(
                    x=["Expected Return (%)", "Volatility (%)"],
                    y=[row["Expected_Return"].values[0]*100, row["Volatility"].values[0]*100],
                    name=profile, marker_color=color
                ))
        fig_comp.update_layout(barmode="group")
        st.plotly_chart(dark_fig(fig_comp, 340), use_container_width=True)

    # Per-profile detail
    if portfolios is not None and len(portfolios):
        st.markdown("<div class='section-header'>Detailed Allocations</div>",
                    unsafe_allow_html=True)

        tab_con, tab_bal, tab_agg = st.tabs(["🟢 Conservative", "🟡 Balanced", "🔴 Aggressive"])

        for tab, profile in [(tab_con, "Conservative"), (tab_bal, "Balanced"), (tab_agg, "Aggressive")]:
            with tab:
                df_p   = portfolios[portfolios["Profile"] == profile].sort_values("Weight_Pct", ascending=False)
                color  = PROFILE_COLORS[profile]

                c_pie, c_bar = st.columns(2)
                with c_pie:
                    fig_pie = go.Figure(go.Pie(
                        labels=df_p["Symbol"], values=df_p["Weight_Pct"],
                        hole=0.4, textinfo="label+percent",
                        marker_colors=px.colors.qualitative.Plotly
                    ))
                    fig_pie.update_layout(showlegend=False)
                    st.plotly_chart(dark_fig(fig_pie, 340), use_container_width=True)

                with c_bar:
                    if "Sector" in df_p.columns:
                        sec_wt = df_p.groupby("Sector")["Weight_Pct"].sum().sort_values()
                        fig_sec = go.Figure(go.Bar(x=sec_wt.values, y=sec_wt.index,
                                                    orientation="h",
                                                    marker_color=color, opacity=0.85))
                        fig_sec.update_layout(xaxis_title="Weight (%)", yaxis_title="")
                        st.plotly_chart(dark_fig(fig_sec, 340), use_container_width=True)

                # Stock table
                disp = df_p[["Symbol", "Sector", "Weight_Pct"]].rename(
                    columns={"Weight_Pct": "Weight (%)"})
                st.dataframe(disp.reset_index(drop=True),
                             use_container_width=True, hide_index=True)

    # ── Custom portfolio simulator
    st.divider()
    st.markdown("<div class='section-header'>Custom Portfolio Simulator</div>",
                unsafe_allow_html=True)
    st.caption("Pick stocks and weights, then see a simulated cumulative return curve.")

    available_syms = sorted(price_raw["Symbol"].unique())
    chosen = st.multiselect("Select stocks", available_syms,
                             default=available_syms[:5] if len(available_syms) >= 5 else available_syms)

    if chosen:
        weights_raw = {}
        cols_w = st.columns(min(len(chosen), 5))
        for i, sym in enumerate(chosen):
            with cols_w[i % 5]:
                weights_raw[sym] = st.number_input(sym, min_value=0.0, max_value=100.0,
                                                    value=round(100 / len(chosen), 1),
                                                    step=0.5, key=f"wt_{sym}")

        total_w = sum(weights_raw.values())
        if total_w > 0:
            weights = {s: w / total_w for s, w in weights_raw.items()}
            pivot   = price_raw[price_raw["Symbol"].isin(chosen)].pivot_table(
                index="Date", columns="Symbol", values="Close")
            ret_mat = pivot.pct_change().dropna()
            port_ret = sum(ret_mat[s] * w for s, w in weights.items() if s in ret_mat.columns)
            cum      = (1 + port_ret).cumprod() * 100

            fig_cust = go.Figure(go.Scatter(x=cum.index, y=cum.values,
                                             line=dict(color=COLORS["accent"], width=2),
                                             fill="tozeroy",
                                             fillcolor="rgba(88,166,255,0.08)"))
            fig_cust.update_layout(yaxis_title="Portfolio Value (base=100)", xaxis_title="")
            st.plotly_chart(dark_fig(fig_cust, 360), use_container_width=True)

            ann_ret = (cum.iloc[-1] / 100) ** (252 / max(len(cum), 1)) - 1
            ann_vol_cust = port_ret.std() * np.sqrt(252)
            sharpe_cust = (ann_ret - 0.06) / ann_vol_cust if ann_vol_cust > 0 else 0
            r1, r2, r3 = st.columns(3)
            for rc, lbl, val in [
                (r1, "Annualised Return", f"{ann_ret*100:.2f}%"),
                (r2, "Annualised Volatility", f"{ann_vol_cust*100:.2f}%"),
                (r3, "Sharpe Ratio", f"{sharpe_cust:.3f}"),
            ]:
                rc.markdown(f"""
                <div class='kpi-card'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value'>{val}</div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RISK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Risk Dashboard":

    st.markdown("# Risk Dashboard")
    st.markdown(f"<span style='color:{COLORS['muted']};font-size:14px'>"
                "Volatility · Sharpe · Sortino · Max Drawdown · VaR · CVaR"
                "</span>", unsafe_allow_html=True)
    st.divider()

    # ── Portfolio-level risk
    if port_risk is not None and len(port_risk):
        st.markdown("<div class='section-header'>Portfolio Risk Metrics</div>",
                    unsafe_allow_html=True)

        cols_pr = st.columns(3)
        for i, profile in enumerate(["Conservative", "Balanced", "Aggressive"]):
            if profile not in port_risk.index:
                continue
            row   = port_risk.loc[profile]
            color = PROFILE_COLORS[profile]
            icon  = {"Conservative": "🟢", "Balanced": "🟡", "Aggressive": "🔴"}[profile]
            with cols_pr[i]:
                st.markdown(f"""
                <div class='kpi-card' style='border-color:{color}'>
                  <div class='kpi-label'>{icon} {profile}</div>
                  <div style='margin-top:8px'>
                    {''.join(f"<div style='display:flex;justify-content:space-between;margin:5px 0'>"
                             f"<span style='color:{COLORS['muted']};font-size:12px'>{k}</span>"
                             f"<span style='font-size:12px;font-weight:600'>{v}</span></div>"
                             for k, v in [
                               ("Annual Return", f"{row.get('Annual_Return', 0):.1f}%"),
                               ("Volatility",    f"{row.get('Volatility', 0):.1f}%"),
                               ("Sharpe Ratio",  f"{row.get('Sharpe_Ratio', 0):.3f}"),
                               ("Sortino Ratio", f"{row.get('Sortino_Ratio', 0):.3f}"),
                               ("Max Drawdown",  f"{row.get('Max_Drawdown', 0):.1f}%"),
                               ("VaR 95%",       f"{row.get('VaR_95', 0):.2f}%"),
                             ])}
                  </div>
                </div>""", unsafe_allow_html=True)

        # Radar chart for portfolios
        categories = ["Annual_Return", "Volatility", "Sharpe_Ratio", "Sortino_Ratio"]
        fig_rad = go.Figure()
        for profile, color in PROFILE_COLORS.items():
            if profile not in port_risk.index:
                continue
            vals = [abs(port_risk.loc[profile, c]) for c in categories]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself", name=profile,
                line_color=color, fillcolor=color.replace("#", "rgba(") + ",0.1)"
                    if color.startswith("#") else color
            ))
        fig_rad.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)"),
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(color=COLORS["text"]),
                              height=380, margin=dict(l=40,r=40,t=40,b=40))
        st.plotly_chart(fig_rad, use_container_width=True)

        # Cumulative returns simulation
        st.markdown("<div class='section-header'>Simulated Portfolio Growth (Full History)</div>",
                    unsafe_allow_html=True)

        if portfolios is not None:
            pivot_all = price_raw.pivot_table(index="Date", columns="Symbol", values="Close")
            ret_all   = pivot_all.pct_change().dropna()
            fig_cum   = go.Figure()

            for profile, color in PROFILE_COLORS.items():
                df_p = portfolios[portfolios["Profile"] == profile].copy()
                df_p["Weight"] = df_p["Weight_Pct"] / 100
                syms_ok = [s for s in df_p["Symbol"] if s in ret_all.columns]
                if not syms_ok:
                    continue
                wts_ok  = df_p[df_p["Symbol"].isin(syms_ok)].set_index("Symbol")["Weight"]
                wts_ok  = wts_ok / wts_ok.sum()
                port_ret = ret_all[syms_ok].dot(wts_ok)
                cum      = (1 + port_ret).cumprod() * 100

                fig_cum.add_trace(go.Scatter(x=cum.index, y=cum.values,
                                              name=profile, line=dict(color=color, width=1.8)))
            fig_cum.update_layout(yaxis_title="Portfolio Value (base=100)", xaxis_title="")
            st.plotly_chart(dark_fig(fig_cum, 400), use_container_width=True)

    # ── Stock-level risk
    if stock_risk is not None and len(stock_risk):
        st.markdown("<div class='section-header'>Individual Stock Risk — Deep Dive</div>",
                    unsafe_allow_html=True)

        sym_sel = st.selectbox("Select stock for drawdown", sorted(stock_risk.index.tolist()), key="risk_sym")

        # Risk vs Return scatter
        sr_plot = stock_risk.reset_index().rename(columns={"index": "Symbol"}) if "Symbol" not in stock_risk.columns else stock_risk.reset_index()
        sr_plot.columns.name = None
        sr_plot["Sector"] = sr_plot.index.map(SECTOR_MAP) if "Symbol" not in sr_plot.columns else sr_plot["Symbol"].map(SECTOR_MAP)

        sr_plot["Sharpe_Size"] = sr_plot["Sharpe_Ratio"].abs()

        fig_rv = px.scatter(sr_plot, x="Volatility", y="Annual_Return",
                             color="Sector", size="Sharpe_Size",
                             hover_name=sr_plot.index if "Symbol" not in sr_plot.columns else "Symbol",
                             labels={"Volatility": "Ann. Volatility (%)",
                                     "Annual_Return": "Ann. Return (%)"},
                             size_max=25)
        fig_rv.add_hline(y=6, line_dash="dash", line_color=COLORS["muted"], line_width=1,
                          annotation_text="Risk-free rate 6%")
        st.plotly_chart(dark_fig(fig_rv, 420), use_container_width=True)

        # Top / Bottom tables
        col_top, col_bot = st.columns(2)
        with col_top:
            st.markdown("**Top 10 by Sharpe Ratio**")
            top_sharpe = stock_risk.sort_values("Sharpe_Ratio", ascending=False).head(10)
            st.dataframe(top_sharpe[["Annual_Return", "Volatility", "Sharpe_Ratio", "Max_Drawdown"]],
                         use_container_width=True)
        with col_bot:
            st.markdown("**Top 10 Riskiest (by Volatility)**")
            top_risk = stock_risk.sort_values("Volatility", ascending=False).head(10)
            st.dataframe(top_risk[["Annual_Return", "Volatility", "Sharpe_Ratio", "Max_Drawdown"]],
                         use_container_width=True)

        # Drawdown chart for selected stock
        if sym_sel in price_raw["Symbol"].values:
            st.markdown(f"<div class='section-header'>Drawdown — {sym_sel}</div>",
                        unsafe_allow_html=True)
            df_dd  = price_raw[price_raw["Symbol"] == sym_sel].sort_values("Date")
            if "Daily_Return" in df_dd.columns:
                cum_r  = (1 + df_dd["Daily_Return"].fillna(0)).cumprod()
                roll_m = cum_r.cummax()
                dd     = (cum_r - roll_m) / roll_m * 100

                fig_dd = go.Figure(go.Scatter(x=df_dd["Date"], y=dd.values,
                                               fill="tozeroy", name="Drawdown",
                                               line=dict(color=COLORS["red"], width=1),
                                               fillcolor="rgba(248,81,73,0.15)"))
                fig_dd.update_layout(yaxis_title="Drawdown (%)", xaxis_title="")
                st.plotly_chart(dark_fig(fig_dd, 320), use_container_width=True)

        # VaR distribution for selected stock
        if sym_sel in price_raw["Symbol"].values and "Daily_Return" in price_raw.columns:
            st.markdown(f"<div class='section-header'>Return Distribution & VaR — {sym_sel}</div>",
                        unsafe_allow_html=True)
            rets    = price_raw[price_raw["Symbol"] == sym_sel]["Daily_Return"].dropna() * 100
            var_95  = np.percentile(rets, 5)
            cvar_95 = rets[rets <= var_95].mean()

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=rets, nbinsx=100, name="Daily Returns",
                                             marker_color=COLORS["accent"], opacity=0.7))
            fig_dist.add_vline(x=var_95, line_dash="dash", line_color=COLORS["red"],
                                annotation_text=f"VaR 95%: {var_95:.2f}%")
            fig_dist.add_vline(x=cvar_95, line_dash="dot", line_color=COLORS["yellow"],
                                annotation_text=f"CVaR: {cvar_95:.2f}%")
            fig_dist.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency")
            st.plotly_chart(dark_fig(fig_dist, 340), use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{COLORS['muted']};font-size:11px;padding:6px 0'>"
    "NIFTY-50 Investment Intelligence Platform · NIFTY-50 Dataset (NSE India, 2000–2021) · "
    "For educational and competition purposes only · Not financial advice"
    "</div>",
    unsafe_allow_html=True,
)
