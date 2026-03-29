import streamlit as st
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os

from api import run_for_city, run_custom
from city_data import list_cities, get_city
from engine import Distributions

# ── Gemini API setup ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Page config ──
st.set_page_config(page_title="RentOrOwn", page_icon="🏠", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

    /* ═══ PALETTE ═══
       bg-deep:   #07090f    (near-black with blue undertone)
       bg-panel:  #0d1117    (GitHub dark)
       bg-card:   #141b27    (elevated surface)
       border:    #1e2a3a    (subtle)
       border-hi: #2a3a4f    (hover)
       text-1:    #f0f4f8    (primary)
       text-2:    #8899aa    (secondary)
       text-3:    #556677    (tertiary)
       buy:       #22d3ee    (cyan - signals "buying")
       rent:      #f59e0b    (amber - signals "renting")
       accent:    #f97316    (orange - CTAs)
       positive:  #34d399    (green)
       negative:  #f87171    (red)
    ═══════════════════ */

    /* ── Global ── */
    .stApp {
        background: #07090f;
        color: #f0f4f8;
        background-image:
            linear-gradient(rgba(30, 42, 58, 0.15) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 42, 58, 0.15) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, p, span, label, div {
        font-family: 'Libre Franklin', -apple-system, sans-serif;
    }
    h1 {
        font-family: 'Libre Franklin', sans-serif !important;
        font-size: 1.8rem !important;
        font-weight: 900 !important;
        color: #f0f4f8 !important;
        letter-spacing: -1px;
        margin-bottom: 0 !important;
        text-transform: uppercase;
    }
    .block-container h1 {
        border-bottom: none;
        display: inline-block;
        padding-bottom: 0;
        position: relative;
    }
    .block-container h1::after {
        content: '';
        position: absolute;
        bottom: -4px;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #f97316);
    }
    h2 {
        font-family: 'Libre Franklin', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        color: #8899aa !important;
        letter-spacing: 3px;
        text-transform: uppercase;
        border-left: 3px solid #22d3ee;
        padding-left: 10px;
        margin-top: 0.5rem !important;
    }
    h3 {
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        color: #f0f4f8 !important;
        letter-spacing: 0.5px;
    }

    /* ── Subtitle ── */
    .subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        color: #556677;
        margin-top: 6px;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    /* ── City badge ── */
    .city-badge {
        display: inline-block;
        background: #141b27;
        color: #f0f4f8;
        font-family: 'Space Mono', monospace;
        font-size: 0.8rem;
        font-weight: 400;
        padding: 8px 16px;
        border: 1px solid #1e2a3a;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    .city-badge .accent {
        color: #f97316;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #0d1117;
        border: 1px solid #1e2a3a;
        border-radius: 0;
        padding: 1rem 1.2rem;
        position: relative;
        overflow: hidden;
    }
    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee 0%, #22d3ee 50%, transparent 50%);
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.6rem !important;
        font-weight: 400 !important;
        color: #556677 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #f0f4f8 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1e2a3a;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f0f4f8 !important;
        border: none;
        padding-left: 0;
        margin-top: 0 !important;
    }
    [data-testid="stSidebar"] label {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem !important;
        font-weight: 400 !important;
        color: #556677 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput input {
        background: #141b27 !important;
        color: #f0f4f8 !important;
        border-color: #1e2a3a !important;
        font-family: 'Space Mono', monospace !important;
        border-radius: 0 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #8899aa !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #f97316 !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #f97316, #ea580c) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.75rem !important;
        padding: 0.8rem 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.15s ease !important;
        box-shadow: 0 0 20px rgba(249, 115, 22, 0.15) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 30px rgba(249, 115, 22, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: #1e2a3a !important;
        margin: 1rem 0 !important;
    }

    /* ── Plotly containers ── */
    [data-testid="stPlotlyChart"] {
        background: #0d1117;
        border: 1px solid #1e2a3a;
        border-radius: 0;
        padding: 0.3rem;
    }

    /* ── Summary box ── */
    .summary-box {
        background: #0d1117;
        border: 1px solid #1e2a3a;
        border-left: 3px solid #22d3ee;
        padding: 1.2rem 1.4rem;
        font-family: 'Libre Franklin', sans-serif;
        font-size: 0.92rem;
        line-height: 1.7;
        color: #f0f4f8;
    }

    /* ── Captions ── */
    .stCaption {
        color: #556677 !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.5px;
    }

    /* ── Footer ── */
    .data-footer {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #556677;
        letter-spacing: 1px;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #1e2a3a;
        margin-top: 2rem;
    }

    .stAlert { border-radius: 0 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ═══ LANDING PAGE ═══ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(25px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; } to { opacity: 1; }
    }
    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(34, 211, 238, 0.1); }
        50% { box-shadow: 0 0 30px 5px rgba(34, 211, 238, 0.08); }
    }
    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }

    .landing-hero {
        text-align: center;
        padding: 3rem 0 1.5rem;
        animation: fadeIn 0.8s ease-out;
        position: relative;
    }
    .landing-hero h2 {
        font-family: 'Libre Franklin', sans-serif !important;
        font-size: 3.2rem !important;
        font-weight: 900 !important;
        color: #f0f4f8 !important;
        letter-spacing: -2px;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.05;
        text-transform: uppercase;
    }
    .hero-accent {
        background: linear-gradient(135deg, #22d3ee, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-accent-rent {
        background: linear-gradient(135deg, #f59e0b, #f97316);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .landing-tagline {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        color: #556677;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }

    .landing-stat-row {
        display: flex;
        justify-content: center;
        gap: 4rem;
        margin: 2rem 0;
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }
    .landing-stat { text-align: center; }
    .landing-stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #22d3ee;
    }
    .landing-stat-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.6rem;
        color: #556677;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 4px;
    }

    .landing-divider {
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #22d3ee, #f97316);
        margin: 2rem auto;
        animation: fadeIn 1s ease-out 0.5s both;
    }

    /* Step cards — dark theme */
    .step-card {
        background: #0d1117;
        border: 1px solid #1e2a3a;
        border-radius: 0;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }
    .step-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, #22d3ee, #f97316);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.4s ease;
    }
    .step-card:hover::before { transform: scaleX(1); }
    .step-card:hover {
        border-color: #2a3a4f;
        box-shadow: 0 0 30px rgba(34, 211, 238, 0.06);
    }
    .step-number {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #22d3ee;
        line-height: 1;
        margin-bottom: 8px;
    }
    .step-title {
        font-family: 'Libre Franklin', sans-serif;
        font-size: 1rem;
        font-weight: 800;
        color: #f0f4f8;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .step-desc {
        font-family: 'Libre Franklin', sans-serif;
        color: #8899aa;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .anim-1 { animation: fadeInUp 0.5s ease-out 0.2s both; }
    .anim-2 { animation: fadeInUp 0.5s ease-out 0.35s both; }
    .anim-3 { animation: fadeInUp 0.5s ease-out 0.5s both; }

    /* City grid */
    .city-grid-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #556677;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-align: center;
        margin: 2rem 0 1rem;
    }
    .city-card {
        background: #0d1117;
        border: 1px solid #1e2a3a;
        border-radius: 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
        cursor: default;
        animation: fadeInUp 0.4s ease-out both;
    }
    .city-card:hover {
        border-left: 3px solid #22d3ee;
        background: #141b27;
    }
    .city-name {
        font-family: 'Libre Franklin', sans-serif;
        font-weight: 700;
        color: #f0f4f8;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }
    .city-price {
        font-family: 'Space Mono', monospace;
        color: #8899aa;
        font-size: 0.72rem;
    }
    .city-tag {
        display: inline-block;
        font-family: 'Space Mono', monospace;
        font-size: 0.55rem;
        font-weight: 700;
        padding: 2px 6px;
        margin-top: 4px;
        letter-spacing: 1px;
    }
    .tag-college {
        background: rgba(34, 211, 238, 0.1);
        color: #22d3ee;
        border: 1px solid rgba(34, 211, 238, 0.2);
    }
    .tag-metro {
        background: rgba(249, 115, 22, 0.1);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.2);
    }

    .cta-hint {
        text-align: center;
        margin-top: 2rem;
        animation: fadeIn 1s ease-out 1s both;
    }
    .cta-hint span {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #556677;
        letter-spacing: 0.5px;
    }
    .cta-arrow {
        display: inline-block;
        font-size: 1.1rem;
        color: #22d3ee;
        animation: glowPulse 2.5s infinite;
        margin-left: 6px;
    }

    /* ── Chat ── */
    [data-testid="stChatMessage"] {
        background: #0d1117 !important;
        border: 1px solid #1e2a3a !important;
        border-radius: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("RentOrOwn")
st.markdown('<p class="subtitle">Every rent vs. buy calculator gives you one number. We give you 10,000.</p>', unsafe_allow_html=True)


# SIDEBAR INPUTS

with st.sidebar:
    st.header("Your Scenario")

    # City selector
    cities = list_cities()
    city_names = ["Custom"] + [c["name"] for c in cities]
    city_keys = ["custom"] + [c["key"] for c in cities]
    selected_city_name = st.selectbox("City", city_names, index=1)  # Default: State College
    selected_city_key = city_keys[city_names.index(selected_city_name)]

    # If city selected, pre-fill values
    if selected_city_key != "custom":
        from city_data import get_city
        city_info = get_city(selected_city_key)
        default_price = city_info["median_home_price"]
        default_rent = city_info["median_rent"]
    else:
        default_price = 300_000
        default_rent = 1_500

    home_price = st.number_input(
        "Home Price ($)", min_value=50_000, max_value=3_000_000,
        value=default_price, step=10_000,
    )

    down_payment_pct = st.slider("Down Payment (%)", 0, 100, 20, 1)

    mortgage_rate = st.number_input(
        "Mortgage Rate (%)", min_value=0.0, max_value=15.0,
        value=7.0, step=0.1,
    )

    mortgage_term = st.slider(
        "Mortgage Term (years)",
        min_value=5, max_value=40, value=30, step=1,
    )

    monthly_rent_input = st.number_input(
        "Monthly Rent ($)", min_value=200, max_value=20_000,
        value=default_rent, step=100,
        help="City mode will use this instead of the preset median."
    )

    st.markdown("**What will you do with the savings?**")
    strategy_label = st.selectbox(
        "Savings discipline",
        ["Nothing (0%)", "High-yield savings (~4.5%)", "Invest in index funds (~10%)"],
        index=1,
        help="Models how the renter deploys surplus cash and the initial down payment if they rent."
    )

    strategy_map = {
        "Nothing (0%)": {"mean": 0.0, "std": 0.0001},
        "High-yield savings (~4.5%)": {"mean": 0.045, "std": 0.01},
        "Invest in index funds (~10%)": {"mean": 0.10, "std": 0.16},
    }
    strategy = strategy_map[strategy_label]

    buyer_invest = st.checkbox(
        "Buyer invests surplus cash", value=True,
        help=(
            "When on: buyer monthly surplus and post-payoff mortgage dollars are invested at the stock return draw. "
            "When off: buyer surplus stays as idle cash (no growth)."
        ),
    )
    renter_invest = st.checkbox(
        "Renter invests surplus cash", value=True,
        help=(
            "When on: renter monthly savings and the upfront parity cash (down payment + closing) are invested at the stock return draw. "
            "When off: they remain as idle cash (no growth)."
        ),
    )

    time_horizon = st.slider(
        "How long do you plan to stay? (years)",
        min_value=1, max_value=50, value=7,
        help="This is the single most important variable",
    )

    st.markdown("---")
    run_button = st.button("Run 10,000 Simulations", type="primary", use_container_width=True)

    # ── Compare Mode ──
    st.markdown("---")
    st.header("Compare Two Cities")
    compare_city_names = [c["name"] for c in cities]
    compare_city_keys = [c["key"] for c in cities]

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        compare_a_name = st.selectbox("City A", compare_city_names, index=0, key="comp_a")
    with comp_col2:
        compare_b_name = st.selectbox("City B", compare_city_names, index=8, key="comp_b")  # NYC default

    compare_horizon = st.slider("Compare Horizon (years)", 1, 30, 10, key="comp_horizon")
    compare_button = st.button("Compare Cities", use_container_width=True)



# CHART BUILDERS

def build_histogram(results):
    """Histogram of buy vs rent final wealth distributions."""
    wealth_diff = np.array(results["buy_wealth_distribution"]) - np.array(results["rent_wealth_distribution"])
    buy_wins = wealth_diff[wealth_diff >= 0]
    rent_wins = wealth_diff[wealth_diff < 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=buy_wins, nbinsx=60, name="Buying Wins",
        marker_color="#22d3ee", opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=rent_wins, nbinsx=60, name="Renting Wins",
        marker_color="#f97316", opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#f0f4f8", line_width=2)
    fig.update_layout(
        title=dict(text="DISTRIBUTION OF OUTCOMES", font=dict(family="Libre Franklin, sans-serif", size=13, color="#f0f4f8")),
        xaxis_title="Net Wealth Difference (Buy − Rent)",
        yaxis_title="Simulations",
        barmode="overlay", height=400,
        paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        font=dict(family="Libre Franklin, sans-serif", size=11, color="#f0f4f8"),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        xaxis=dict(tickformat="$,.0f", gridcolor="#1e2a3a", zerolinecolor="#f0f4f8"),
        yaxis=dict(gridcolor="#1e2a3a"),
        margin=dict(t=40, b=40, l=50, r=20),
    )
    return fig


def build_wealth_over_time(results):
    """Median wealth paths with 25-75% confidence bands."""
    years = results["years"]

    fig = go.Figure()

    # Buy confidence band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=results["buy_p75_trajectory"] + results["buy_p25_trajectory"][::-1],
        fill="toself", fillcolor="rgba(34, 211, 238, 0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Rent confidence band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=results["rent_p75_trajectory"] + results["rent_p25_trajectory"][::-1],
        fill="toself", fillcolor="rgba(249, 115, 22, 0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Median lines
    fig.add_trace(go.Scatter(
        x=years, y=results["buy_median_trajectory"],
        mode="lines", name="Buy (Median)",
        line=dict(color="#22d3ee", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=results["rent_median_trajectory"],
        mode="lines", name="Rent + Save (Median)",
        line=dict(color="#f97316", width=3),
    ))

    fig.update_layout(
        title=dict(text="WEALTH OVER TIME", font=dict(family="Libre Franklin, sans-serif", size=13, color="#f0f4f8")),
        xaxis_title="Year", yaxis_title="Net Wealth ($)",
        height=400,
        paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        font=dict(family="Libre Franklin, sans-serif", size=11, color="#f0f4f8"),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        yaxis=dict(tickformat="$,.0f", gridcolor="#1e2a3a"),
        xaxis=dict(gridcolor="#1e2a3a"),
        margin=dict(t=40, b=40, l=60, r=20),
    )
    return fig


def build_sensitivity_chart(sensitivity):
    """Tornado chart of variable correlations with outcome."""
    # Sort by absolute correlation
    sorted_vars = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)

    # Clean up names for display
    display_names = {
        "home_appreciation": "Home Appreciation",
        "rent_inflation": "Rent Inflation",
        "stock_returns": "S&P 500 Returns",
        "maintenance_costs": "Maintenance Costs",
    }

    names = [display_names.get(s[0], s[0]) for s in sorted_vars]
    values = [s[1] for s in sorted_vars]
    colors = ["#0f2b46" if v > 0 else "#e8590c" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="Space Mono, monospace", size=11, color="#1a2a3a"),
    ))
    fig.update_layout(
        title=dict(text="SENSITIVITY ANALYSIS", font=dict(family="Libre Franklin, sans-serif", size=13, color="#f0f4f8")),
        xaxis_title="Correlation with Outcome",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Libre Franklin, sans-serif", size=11, color="#f0f4f8"),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="#f0f4f8", zerolinewidth=1,
               gridcolor="#1e2a3a"),
        margin=dict(t=40, b=40, l=120, r=20),
    )
    return fig


# GEMINI SUMMARY

def get_gemini_summary(results, home_price, monthly_rent, down_payment_pct,
                       mortgage_rate, time_horizon, selected_city_name):
    """Call Gemini API and return summary text. Returns None on failure."""
    prompt = f"""You are a financial analyst explaining Monte Carlo simulation results to a 25-year-old considering their first home purchase. Be direct, specific, and concise. No fluff. Use the exact numbers provided.

Simulation inputs:
- City: {selected_city_name}
- Home price: ${home_price:,.0f}
- Down payment: {down_payment_pct}%
- Mortgage rate: {mortgage_rate}%
- Monthly rent alternative: ${monthly_rent:,.0f}
- Time horizon: {time_horizon} years

Results from 10,000 Monte Carlo simulations:
- Buying wins in {results['buy_wins_pct']:.1f}% of simulations
- Break-even year: {results['breakeven_year'] if results['breakeven_year'] else 'Never within the time horizon'}
- Median wealth if you buy: ${results['median_buy']:,.0f}
- Median wealth if you rent + invest: ${results['median_rent']:,.0f}
- Median advantage: ${results['median_advantage']:,.0f} ({'favoring buying' if results['median_advantage'] > 0 else 'favoring renting'})
- 10th percentile outcome: ${results['p10_advantage']:,.0f}
- 90th percentile outcome: ${results['p90_advantage']:,.0f}

Write exactly 4 sentences:
1. The verdict — who wins and by how much.
2. The break-even insight — when buying overtakes renting.
3. The risk range — what happens in bad vs good scenarios.
4. One specific actionable takeaway.

Do NOT include disclaimers or caveats."""

    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set -- skipping AI summary")
        return None

    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[WARN] Gemini {model_name} failed: {e}")
            continue
    return None


def build_chat_context(results, home_price, monthly_rent, down_payment_pct, mortgage_rate,
                       mortgage_term, time_horizon, strategy_label, selected_city_name):
    """Flatten the simulation results into a concise context string for chat."""
    be = results.get("breakeven_year")
    sensitivity = results.get("sensitivity", {})
    top_sensitivity = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)
    top_sensitivity_str = ", ".join([f"{k}: {v:+.3f}" for k, v in top_sensitivity[:4]])

    return (
        f"City: {selected_city_name}; Home price: ${home_price:,.0f}; "
        f"Down payment: {down_payment_pct}%; Rate: {mortgage_rate}%; Term: {mortgage_term} years; "
        f"Rent alternative: ${monthly_rent:,.0f}; Horizon: {time_horizon} years; Savings plan: {strategy_label}. "
        f"Buying wins {results.get('buy_wins_pct', 0):.1f}% of simulations; "
        f"Median buy wealth ${results.get('median_buy', 0):,.0f}; Median rent wealth ${results.get('median_rent', 0):,.0f}; "
        f"Median advantage ${results.get('median_advantage', 0):,.0f}; P10 advantage ${results.get('p10_advantage', 0):,.0f}; "
        f"P90 advantage ${results.get('p90_advantage', 0):,.0f}; Break-even year: {be if be else 'None in horizon'}. "
        f"Top sensitivities: {top_sensitivity_str if top_sensitivity_str else 'n/a'}."
    )


def get_gemini_chat_reply(user_message, chat_history, context_text):
    """Chat-style Gemini response grounded in the current scenario."""
    history_lines = []
    for msg in chat_history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        history_lines.append(f"{speaker}: {msg['content']}")
    history_block = "\n".join(history_lines) if history_lines else "None yet."

    prompt = f"""You are a financial analyst helping someone interpret their rent vs buy simulation.
Scenario context (use these numbers): {context_text}

Conversation so far:
{history_block}

Latest user question: {user_message}

Reply with 3-5 sentences, cite key numbers, and stay anchored to this scenario. If asked beyond the provided data, say you only have this scenario's results."""

    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set -- skipping chat reply")
        return None

    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"[WARN] Gemini chat {model_name} failed: {e}")
            continue
    return None



    # MAIN DISPLAY

# Build a key for the current sidebar settings
current_scenario_key = f"{selected_city_key}|{home_price}|{monthly_rent_input}|{down_payment_pct}|{mortgage_rate}|{mortgage_term}|{time_horizon}|{strategy_label}|{buyer_invest}|{renter_invest}"

results = None
compare_active = False

# COMPARE MODE

if compare_button:
    compare_active = True
    key_a = compare_city_keys[compare_city_names.index(compare_a_name)]
    key_b = compare_city_keys[compare_city_names.index(compare_b_name)]
    info_a = get_city(key_a)
    info_b = get_city(key_b)

    dist = Distributions(stock_return_mean=strategy["mean"], stock_return_std=strategy["std"])

    with st.spinner(f"Simulating {compare_a_name} vs {compare_b_name}..."):
        results_a = run_for_city(
            city_key=key_a, down_payment_pct=down_payment_pct / 100,
            mortgage_rate=mortgage_rate / 100, mortgage_term_years=mortgage_term,
            time_horizon_years=compare_horizon, distributions=dist,
            invest_surplus=False, buyer_invest_surplus=buyer_invest, renter_invest_surplus=renter_invest,
        )
        results_b = run_for_city(
            city_key=key_b, down_payment_pct=down_payment_pct / 100,
            mortgage_rate=mortgage_rate / 100, mortgage_term_years=mortgage_term,
            time_horizon_years=compare_horizon, distributions=dist,
            invest_surplus=False, buyer_invest_surplus=buyer_invest, renter_invest_surplus=renter_invest,
        )

    st.markdown("## City Comparison")
    st.markdown(f"Both at **{compare_horizon}-year** horizon · **{down_payment_pct}%** down · **{mortgage_rate}%** rate")
    st.markdown("---")

    # Side by side metrics
    col_left, col_divider, col_right = st.columns([5, 1, 5])

    with col_left:
        st.markdown(f"### {compare_a_name}")
        st.markdown(f"`${info_a['median_home_price']:,.0f} home · ${info_a['median_rent']:,.0f}/mo rent`")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Buy Wins", f"{results_a['buy_wins_pct']:.0f}%")
        with m2:
            be_a = results_a["breakeven_year"]
            st.metric("Break-Even", f"Yr {be_a}" if be_a else "Never")
        with m3:
            med_a = results_a["median_advantage"]
            st.metric("Median Edge", f"${med_a:,.0f}")
        st.plotly_chart(build_wealth_over_time(results_a), use_container_width=True)

    with col_divider:
        st.markdown("<div style='border-left: 2px solid #e1e5eb; height: 500px; margin: 0 auto; width: 0;'></div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(f"### {compare_b_name}")
        st.markdown(f"`${info_b['median_home_price']:,.0f} home · ${info_b['median_rent']:,.0f}/mo rent`")
        m4, m5, m6 = st.columns(3)
        with m4:
            st.metric("Buy Wins", f"{results_b['buy_wins_pct']:.0f}%")
        with m5:
            be_b = results_b["breakeven_year"]
            st.metric("Break-Even", f"Yr {be_b}" if be_b else "Never")
        with m6:
            med_b = results_b["median_advantage"]
            st.metric("Median Edge", f"${med_b:,.0f}")
        st.plotly_chart(build_wealth_over_time(results_b), use_container_width=True)

    # Head-to-head bar chart
    st.markdown("---")
    fig_compare = go.Figure()
    categories = ["Buy Win %", "Median Buy Wealth", "Median Rent Wealth"]
    fig_compare.add_trace(go.Bar(
        name=compare_a_name,
        x=categories,
        y=[results_a["buy_wins_pct"], results_a["median_buy"], results_a["median_rent"]],
        marker_color="#22d3ee",
        text=[f"{results_a['buy_wins_pct']:.0f}%", f"${results_a['median_buy']:,.0f}", f"${results_a['median_rent']:,.0f}"],
        textposition="outside",
    ))
    fig_compare.add_trace(go.Bar(
        name=compare_b_name,
        x=categories,
        y=[results_b["buy_wins_pct"], results_b["median_buy"], results_b["median_rent"]],
        marker_color="#f97316",
        text=[f"{results_b['buy_wins_pct']:.0f}%", f"${results_b['median_buy']:,.0f}", f"${results_b['median_rent']:,.0f}"],
        textposition="outside",
    ))
    fig_compare.update_layout(
        title=dict(text="HEAD TO HEAD", font=dict(family="Libre Franklin, sans-serif", size=14, color="#f0f4f8")),
        barmode="group", height=400,
        paper_bgcolor="#07090f", plot_bgcolor="#0d1117",
        font=dict(family="Libre Franklin, sans-serif", size=11, color="#f0f4f8"),
        yaxis=dict(gridcolor="#1e2a3a", tickformat=",.0f"),
        legend=dict(font=dict(size=12)),
    )
    st.plotly_chart(fig_compare, use_container_width=True)

elif run_button:
    with st.spinner("Running 10,000 simulations..."):
        dist = Distributions(stock_return_mean=strategy["mean"], stock_return_std=strategy["std"])
        if selected_city_key != "custom":
            results = run_for_city(
                city_key=selected_city_key,
                down_payment_pct=down_payment_pct / 100,
                mortgage_rate=mortgage_rate / 100,
                mortgage_term_years=mortgage_term,
                time_horizon_years=time_horizon,
                distributions=dist,
                invest_surplus=False,
                buyer_invest_surplus=buyer_invest,
                renter_invest_surplus=renter_invest,
                monthly_rent_override=monthly_rent_input,
            )
        else:
            results = run_custom(
                home_price=home_price,
                monthly_rent=monthly_rent_input,
                down_payment_pct=down_payment_pct / 100,
                mortgage_rate=mortgage_rate / 100,
                mortgage_term_years=mortgage_term,
                time_horizon_years=time_horizon,
                distributions=dist,
                invest_surplus=False,
                buyer_invest_surplus=buyer_invest,
                renter_invest_surplus=renter_invest,
            )

    # Cache the latest run so chat interactions don't drop back to landing
    st.session_state["last_results"] = results
    st.session_state["last_meta"] = {
        "home_price": home_price,
        "monthly_rent": monthly_rent_input,
        "down_payment_pct": down_payment_pct,
        "mortgage_rate": mortgage_rate,
        "mortgage_term": mortgage_term,
        "time_horizon": time_horizon,
        "strategy_label": strategy_label,
        "buyer_invest": buyer_invest,
        "renter_invest": renter_invest,
        "selected_city_name": selected_city_name,
        "selected_city_key": selected_city_key,
    }
    st.session_state["last_scenario_key"] = current_scenario_key
elif (
    st.session_state.get("last_results")
    and st.session_state.get("last_scenario_key") == current_scenario_key
):
    results = st.session_state["last_results"]


if results is not None:
    # Reset chat when scenario changes and store the current context
    if st.session_state.get("scenario_key") != current_scenario_key:
        st.session_state["scenario_key"] = current_scenario_key
        st.session_state["chat_history"] = []

    st.session_state["chat_context"] = build_chat_context(
        results,
        home_price,
        monthly_rent_input,
        down_payment_pct,
        mortgage_rate,
        mortgage_term,
        time_horizon,
        strategy_label,
        selected_city_name,
    )

    # ── Key metrics ──
    st.markdown("## Results")
    if selected_city_key != "custom":
        st.markdown(f'<div class="city-badge">{results["city"]} <span class="accent">|</span> ${results["home_price"]:,.0f} home <span class="accent">·</span> ${results["monthly_rent"]:,.0f}/mo rent</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Buying Wins In", f"{results['buy_wins_pct']:.0f}% of sims")
    with col2:
        be = results["breakeven_year"]
        st.metric("Break-Even Year", f"~{be} years" if be else "Never")
    with col3:
        med = results["median_advantage"]
        winner = "Buying" if med > 0 else "Renting"
        st.metric(f"Median Edge ({winner})", f"${abs(med):,.0f}")
    with col4:
        st.metric("Downside (10th pctl)", f"${results['p10_advantage']:,.0f}")

    st.markdown("---")

    # ── Charts row 1 ──
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(build_histogram(results), use_container_width=True)
    with c2:
        st.plotly_chart(build_wealth_over_time(results), use_container_width=True)

    st.markdown("---")

    # ── Charts row 2: Sensitivity + AI Summary ──
    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(build_sensitivity_chart(results["sensitivity"]), use_container_width=True)

    with s2:
        st.subheader("Plain English Summary")

        # Try Gemini
        summary = get_gemini_summary(
            results, home_price, monthly_rent_input, down_payment_pct,
            mortgage_rate, time_horizon, selected_city_name
        )

        if summary:
            st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
        else:
            # Fallback
            pct = results["buy_wins_pct"]
            if pct > 60:
                fallback = f"Buying looks favorable — it wins in <strong>{pct:.0f}%</strong> of simulations."
            elif pct > 40:
                fallback = f"It's close to a coin flip. Buying wins <strong>{pct:.0f}%</strong> of the time."
            else:
                fallback = f"Renting and saving looks stronger. Buying only wins <strong>{pct:.0f}%</strong> of the time."

            be = results["breakeven_year"]
            if be:
                fallback += f" Buying breaks even around <strong>year {be}</strong>."
            else:
                fallback += " Buying doesn't break even in your time horizon."

            st.markdown(f'<div class="summary-box">{fallback}</div>', unsafe_allow_html=True)

        st.markdown("")
        st.caption("AI summary powered by Google Gemini · Not financial advice · Uses stochastic assumptions informed by long-run housing, rent, and equity data")

        st.markdown("---")
        st.subheader("Chat About Your Results")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                st.markdown(msg["content"])

        user_question = st.chat_input("Ask a follow-up about this scenario")
        if user_question:
            st.session_state["chat_history"].append({"role": "user", "content": user_question})
            with st.spinner("Gemini is thinking..."):
                reply = get_gemini_chat_reply(
                    user_question,
                    st.session_state["chat_history"],
                    st.session_state.get("chat_context", ""),
                )
            if reply:
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.error("Gemini is unavailable right now. Please try again in a moment.")

elif not compare_active:
    # ── Animated Landing Page ──

    # Hero section
    st.markdown("""
    <div class="landing-hero">
        <h2>Should you <span class="hero-accent-rent">rent</span> or <span class="hero-accent">buy</span>?</h2>
        <div class="landing-tagline">Stop guessing. Start simulating.</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    st.markdown("""
    <div class="landing-stat-row">
        <div class="landing-stat">
            <div class="landing-stat-number">10,000</div>
            <div class="landing-stat-label">Simulations</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-number">14</div>
            <div class="landing-stat-label">Cities</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-number">50</div>
            <div class="landing-stat-label">Years of Data</div>
        </div>
        <div class="landing-stat">
            <div class="landing-stat-number">4</div>
            <div class="landing-stat-label">Risk Variables</div>
        </div>
    </div>
    <div class="landing-divider"></div>
    """, unsafe_allow_html=True)

    # How it works — 3 step cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card anim-1">
            <div class="step-number">01</div>
            <div class="step-title">Pick a City</div>
            <div class="step-desc">14 cities with real median home prices and rents. Or go fully custom.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card anim-2">
            <div class="step-number">02</div>
            <div class="step-title">We Simulate</div>
            <div class="step-desc">10,000 Monte Carlo runs. Each randomizes appreciation, inflation, and costs.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card anim-3">
            <div class="step-number">03</div>
            <div class="step-title">See the Odds</div>
            <div class="step-desc">Full probability distribution. Break-even year. Sensitivity analysis. AI summary.</div>
        </div>
        """, unsafe_allow_html=True)

    # City grid
    st.markdown('<div class="city-grid-title">Available Markets</div>', unsafe_allow_html=True)

    cities_data = list_cities()
    college_towns = ["state_college_pa", "ann_arbor_mi", "chapel_hill_nc", "ames_ia", "austin_tx", "madison_wi", "boulder_co", "tucson_az"]

    city_cols = st.columns(4)
    for i, city in enumerate(cities_data):
        from city_data import get_city
        info = get_city(city["key"])
        is_college = city["key"] in college_towns
        tag_class = "tag-college" if is_college else "tag-metro"
        tag_text = "COLLEGE" if is_college else "METRO"
        delay = 0.8 + (i * 0.06)
        with city_cols[i % 4]:
            st.markdown(f"""
            <div class="city-card" style="animation-delay: {delay}s">
                <div class="city-name">{info['name']}</div>
                <div class="city-price">${info['median_home_price']:,.0f} · ${info['median_rent']:,.0f}/mo</div>
                <span class="city-tag {tag_class}">{tag_text}</span>
            </div>
            """, unsafe_allow_html=True)

    # CTA hint
    st.markdown("""
    <div class="cta-hint">
        <span>Select a city in the sidebar and hit simulate</span>
        <span class="cta-arrow">→</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="data-footer">S&P 500 Total Returns · FHFA + NAR + Shiller Home Prices · BLS CPI Rent Inflation · Google Gemini AI</div>', unsafe_allow_html=True)