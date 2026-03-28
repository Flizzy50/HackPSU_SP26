import streamlit as st
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os

from api import run_for_city, run_custom
from city_data import list_cities
from engine import Distributions

# ── Gemini API setup ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAcCsmpD3ssEpU7a-sWZCSiHvpcm9Z97Vc")
genai.configure(api_key=GEMINI_API_KEY)

# ── Page config ──
st.set_page_config(page_title="RentOrOwn", page_icon="🏠", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ── Global ── */
    .stApp {
        background: #f8f9fb;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        max-width: 1200px;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, p, span, label, div {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    h1 {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #0f2b46 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0 !important;
    }
    .block-container h1 {
        border-bottom: 3px solid #e8590c;
        display: inline-block;
        padding-bottom: 4px;
    }
    h2 {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #0f2b46 !important;
        letter-spacing: -0.3px;
        text-transform: uppercase;
        border-left: 4px solid #e8590c;
        padding-left: 12px;
        margin-top: 0.5rem !important;
    }
    h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #1a3a5c !important;
    }

    /* ── Subtitle ── */
    .subtitle {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        color: #5a6f85;
        margin-top: 2px;
        margin-bottom: 1.5rem;
    }

    /* ── City badge ── */
    .city-badge {
        display: inline-block;
        background: #0f2b46;
        color: #ffffff;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 6px 14px;
        border-radius: 4px;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
    }
    .city-badge .accent {
        color: #e8590c;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-top: 3px solid #0f2b46;
        border-radius: 2px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #5a6f85 !important;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #0f2b46 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0f2b46;
        border-right: none;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        border: none;
        padding-left: 0;
        margin-top: 0 !important;
    }
    [data-testid="stSidebar"] label {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: rgba(255,255,255,0.6) !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput input {
        background: rgba(255,255,255,0.08) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.15) !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: rgba(255,255,255,0.7) !important;
    }
    /* Slider thumb + track */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #e8590c !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[data-testid="stTickBarMax"] {
        color: rgba(255,255,255,0.4) !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #e8590c !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: #e8590c !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 4px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        padding: 0.75rem 1.5rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #c84a0a !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: #e1e5eb !important;
        margin: 1rem 0 !important;
    }

    /* ── Plotly containers ── */
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 2px;
        padding: 0.3rem;
    }

    /* ── Summary box ── */
    .summary-box {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-left: 4px solid #e8590c;
        padding: 1.2rem 1.4rem;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1a3a5c;
    }

    /* ── Landing: steps ── */
    .step-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #e8590c;
        line-height: 1;
        margin-bottom: 6px;
    }
    .step-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #0f2b46;
        margin-bottom: 4px;
    }
    .step-desc {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #5a6f85;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* ── City cards ── */
    .city-card {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 2px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        transition: border-color 0.2s ease;
    }
    .city-card:hover {
        border-left: 3px solid #e8590c;
    }
    .city-name {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: #0f2b46;
        font-size: 0.9rem;
        margin-bottom: 2px;
    }
    .city-price {
        font-family: 'IBM Plex Mono', monospace;
        color: #5a6f85;
        font-size: 0.8rem;
    }

    /* ── Footer ── */
    .data-footer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #8e9baa;
        letter-spacing: 0.5px;
        text-align: center;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid #e1e5eb;
        margin-top: 2rem;
    }

    /* ── Captions ── */
    .stCaption {
        color: #8e9baa !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.75rem !important;
    }

    /* ── Tabs/alerts ── */
    .stAlert { border-radius: 2px !important; }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── LANDING PAGE ANIMATIONS ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-40px); }
        to { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(232, 89, 12, 0.2); }
        50% { box-shadow: 0 0 20px 5px rgba(232, 89, 12, 0.15); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    @keyframes shimmer {
        0% { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* Hero section */
    .landing-hero {
        text-align: center;
        padding: 2rem 0 1.5rem;
        animation: fadeIn 0.8s ease-out;
    }
    .landing-hero h2 {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #0f2b46 !important;
        letter-spacing: -1.5px;
        border: none !important;
        padding: 0 !important;
        margin-bottom: 0.3rem !important;
        line-height: 1.1;
    }
    .landing-hero .hero-accent {
        color: #e8590c;
    }
    .landing-tagline {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 1.15rem;
        color: #5a6f85;
        margin-bottom: 0.5rem;
    }
    .landing-stat-row {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 1.5rem 0;
        animation: fadeInUp 0.8s ease-out 0.3s both;
    }
    .landing-stat {
        text-align: center;
    }
    .landing-stat-number {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #0f2b46;
        animation: countUp 0.6s ease-out 0.5s both;
    }
    .landing-stat-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        color: #8e9baa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 2px;
    }

    /* Divider line */
    .landing-divider {
        width: 60px;
        height: 3px;
        background: #e8590c;
        margin: 1.5rem auto;
        animation: fadeIn 1s ease-out 0.5s both;
    }

    /* Steps row */
    .step-card {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 2px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .step-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #0f2b46, #e8590c);
        transform: scaleX(0);
        transform-origin: left;
        transition: transform 0.4s ease;
    }
    .step-card:hover::before {
        transform: scaleX(1);
    }
    .step-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(15, 43, 70, 0.08);
    }
    .anim-1 { animation: fadeInUp 0.6s ease-out 0.2s both; }
    .anim-2 { animation: fadeInUp 0.6s ease-out 0.4s both; }
    .anim-3 { animation: fadeInUp 0.6s ease-out 0.6s both; }

    /* City grid */
    .city-grid-title {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #8e9baa;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin: 1.5rem 0 1rem;
        animation: fadeIn 0.8s ease-out 0.8s both;
    }
    .city-card {
        background: #ffffff;
        border: 1px solid #e1e5eb;
        border-radius: 2px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.25s ease;
        cursor: default;
        animation: fadeInUp 0.5s ease-out both;
    }
    .city-card:hover {
        border-left: 3px solid #e8590c;
        background: #fafbfc;
        transform: translateX(4px);
    }
    .city-name {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        color: #0f2b46;
        font-size: 0.88rem;
        margin-bottom: 2px;
    }
    .city-price {
        font-family: 'IBM Plex Mono', monospace;
        color: #5a6f85;
        font-size: 0.78rem;
    }
    .city-tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 2px;
        margin-top: 4px;
    }
    .tag-college {
        background: #eef2f7;
        color: #0f2b46;
    }
    .tag-metro {
        background: #fef0e7;
        color: #e8590c;
    }

    /* CTA arrow */
    .cta-hint {
        text-align: center;
        margin-top: 1.5rem;
        animation: fadeIn 1s ease-out 1s both;
    }
    .cta-hint span {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85rem;
        color: #8e9baa;
    }
    .cta-arrow {
        display: inline-block;
        font-size: 1.2rem;
        color: #e8590c;
        animation: pulse 2s infinite;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.title("RentOrOwn")
st.markdown('<p class="subtitle">Every rent vs. buy calculator gives you one number. We give you 10,000.</p>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR INPUTS
# ──────────────────────────────────────────────
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

    mortgage_term = st.selectbox("Mortgage Term", [15, 30], index=1)

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

    time_horizon = st.slider(
        "How long do you plan to stay? (years)",
        min_value=1, max_value=30, value=7,
        help="This is the single most important variable",
    )

    st.markdown("---")
    run_button = st.button("Run 10,000 Simulations", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────
def build_histogram(results):
    """Histogram of buy vs rent final wealth distributions."""
    wealth_diff = np.array(results["buy_wealth_distribution"]) - np.array(results["rent_wealth_distribution"])
    buy_wins = wealth_diff[wealth_diff >= 0]
    rent_wins = wealth_diff[wealth_diff < 0]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=buy_wins, nbinsx=60, name="Buying Wins",
        marker_color="#0f2b46", opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=rent_wins, nbinsx=60, name="Renting Wins",
        marker_color="#e8590c", opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#0f2b46", line_width=2)
    fig.update_layout(
        title=dict(text="DISTRIBUTION OF OUTCOMES", font=dict(family="IBM Plex Sans, Arial, sans-serif", size=13, color="#1a2a3a")),
        xaxis_title="Net Wealth Difference (Buy − Rent)",
        yaxis_title="Simulations",
        barmode="overlay", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Arial, sans-serif", size=11, color="#1a2a3a"),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        xaxis=dict(tickformat="$,.0f", gridcolor="#e1e5eb", zerolinecolor="#0f2b46"),
        yaxis=dict(gridcolor="#e1e5eb"),
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
        fill="toself", fillcolor="rgba(15, 43, 70, 0.1)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Rent confidence band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=results["rent_p75_trajectory"] + results["rent_p25_trajectory"][::-1],
        fill="toself", fillcolor="rgba(232, 89, 12, 0.1)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Median lines
    fig.add_trace(go.Scatter(
        x=years, y=results["buy_median_trajectory"],
        mode="lines", name="Buy (Median)",
        line=dict(color="#0f2b46", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=results["rent_median_trajectory"],
        mode="lines", name="Rent + Save (Median)",
        line=dict(color="#e8590c", width=3),
    ))

    fig.update_layout(
        title=dict(text="WEALTH OVER TIME", font=dict(family="IBM Plex Sans, Arial, sans-serif", size=13, color="#1a2a3a")),
        xaxis_title="Year", yaxis_title="Net Wealth ($)",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Arial, sans-serif", size=11, color="#000000"),
        legend=dict(x=0.02, y=0.98, font=dict(size=10)),
        yaxis=dict(tickformat="$,.0f", gridcolor="#e1e5eb"),
        xaxis=dict(gridcolor="#e1e5eb"),
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
        textfont=dict(family="IBM Plex Mono, Courier New, monospace", size=11, color="#1a2a3a"),
    ))
    fig.update_layout(
        title=dict(text="SENSITIVITY ANALYSIS", font=dict(family="IBM Plex Sans, Arial, sans-serif", size=13, color="#1a2a3a")),
        xaxis_title="Correlation with Outcome",
        height=380,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Arial, sans-serif", size=11, color="#1a2a3a"),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="#0f2b46", zerolinewidth=1,
                   gridcolor="#e1e5eb"),
        margin=dict(t=40, b=40, l=120, r=20),
    )
    return fig


# ──────────────────────────────────────────────
# GEMINI SUMMARY
# ──────────────────────────────────────────────
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

    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text
        except Exception:
            continue
    return None


# ──────────────────────────────────────────────
# MAIN DISPLAY
# ──────────────────────────────────────────────
if run_button:
    with st.spinner("Running 10,000 simulations..."):
        dist = Distributions(stock_return_mean=strategy["mean"], stock_return_std=strategy["std"])
        invest_flag = True  # always apply strategy; 0% just behaves like cash
        if selected_city_key != "custom":
            results = run_for_city(
                city_key=selected_city_key,
                down_payment_pct=down_payment_pct / 100,
                mortgage_rate=mortgage_rate / 100,
                mortgage_term_years=mortgage_term,
                time_horizon_years=time_horizon,
                distributions=dist,
                invest_surplus=invest_flag,
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
                invest_surplus=invest_flag,
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
            results, home_price, monthly_rent, down_payment_pct,
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
        st.caption("AI summary powered by Google Gemini · Not financial advice · Built on historical data from FHFA, BLS, and S&P 500")

else:
    # ── Animated Landing Page ──

    # Hero section
    st.markdown("""
    <div class="landing-hero">
        <h2>Should you <span class="hero-accent">rent</span> or <span class="hero-accent">buy</span>?</h2>
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