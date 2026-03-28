import streamlit as st
import numpy as np
import plotly.graph_objects as go
import google.generativeai as genai
import os

from api import run_for_city, run_custom
from city_data import list_cities

# ── Gemini API setup ──
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAcCsmpD3ssEpU7a-sWZCSiHvpcm9Z97Vc")
genai.configure(api_key=GEMINI_API_KEY)

# ── Page config ──
st.set_page_config(page_title="RentOrOwn", page_icon="🏠", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(160deg, #0a0a0f 0%, #0d1117 40%, #0f1923 100%);
    }

    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Typography */
    h1, h2, h3, .stMetricLabel, .stMetricValue {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hero title */
    h1 {
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d2ff, #7b61ff, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        margin-bottom: 0 !important;
    }

    /* Subtitle styling */
    .subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.15rem;
        color: #8b949e;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(10px);
        transition: border-color 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0, 210, 255, 0.3);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        color: #e6edf3 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(13, 17, 23, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSlider label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #00d2ff, #7b61ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1.5rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 210, 255, 0.35) !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.05) !important;
        margin: 1.5rem 0 !important;
    }

    /* Section headers */
    h2 {
        font-size: 1.8rem !important;
        color: #e6edf3 !important;
        letter-spacing: -0.5px;
    }
    h3 {
        font-size: 1.2rem !important;
        color: #c9d1d9 !important;
    }
    h4 {
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.85rem !important;
    }

    /* Captions */
    .stCaption {
        color: #484f58 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* City cards on landing page */
    .city-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .city-card:hover {
        border-color: rgba(0, 210, 255, 0.3);
        transform: translateY(-2px);
    }
    .city-name {
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        color: #e6edf3;
        font-size: 1rem;
        margin-bottom: 4px;
    }
    .city-price {
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        font-size: 0.85rem;
    }

    /* How it works cards */
    .step-number {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .step-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 0.3rem;
    }
    .step-desc {
        font-family: 'DM Sans', sans-serif;
        color: #8b949e;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Plotly chart containers */
    [data-testid="stPlotlyChart"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 0.5rem;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #00d2ff !important;
    }

    /* Summary section */
    .summary-box {
        background: rgba(0, 210, 255, 0.04);
        border: 1px solid rgba(0, 210, 255, 0.12);
        border-radius: 16px;
        padding: 1.5rem;
    }

    /* Data source footer */
    .data-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #484f58;
        letter-spacing: 0.5px;
        text-align: center;
        padding: 1rem 0;
    }

    /* Warning box styling */
    .stAlert {
        border-radius: 12px !important;
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

    monthly_rent = st.number_input(
        "Monthly Rent ($)", min_value=200, max_value=20_000,
        value=default_rent, step=100,
    )

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
        marker_color="#00d2ff", opacity=0.8,
    ))
    fig.add_trace(go.Histogram(
        x=rent_wins, nbinsx=60, name="Renting Wins",
        marker_color="#ff6b6b", opacity=0.8,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=2)
    fig.update_layout(
        title=dict(text="Distribution of Outcomes", font=dict(family="DM Sans", size=16)),
        xaxis_title="Net Wealth Difference (Buy − Rent)",
        yaxis_title="Simulations",
        barmode="overlay", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b949e"),
        legend=dict(x=0.02, y=0.98, font=dict(size=11)),
        xaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
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
        fill="toself", fillcolor="rgba(0, 210, 255, 0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Rent confidence band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=results["rent_p75_trajectory"] + results["rent_p25_trajectory"][::-1],
        fill="toself", fillcolor="rgba(255, 107, 107, 0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Median lines
    fig.add_trace(go.Scatter(
        x=years, y=results["buy_median_trajectory"],
        mode="lines", name="Buy (Median)",
        line=dict(color="#00d2ff", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=results["rent_median_trajectory"],
        mode="lines", name="Rent + Save (Median)",
        line=dict(color="#ff6b6b", width=3),
    ))

    fig.update_layout(
        title=dict(text="Wealth Over Time", font=dict(family="DM Sans", size=16)),
        xaxis_title="Year", yaxis_title="Net Wealth ($)",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b949e"),
        legend=dict(x=0.02, y=0.98, font=dict(size=11)),
        yaxis=dict(tickformat="$,.0f", gridcolor="rgba(255,255,255,0.04)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
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
    colors = ["#00d2ff" if v > 0 else "#ff6b6b" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12),
    ))
    fig.update_layout(
        title=dict(text="What Matters Most?", font=dict(family="DM Sans", size=16)),
        xaxis_title="Correlation with Outcome",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#8b949e"),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="rgba(255,255,255,0.15)", zerolinewidth=1,
                   gridcolor="rgba(255,255,255,0.04)"),
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
        if selected_city_key != "custom":
            results = run_for_city(
                city_key=selected_city_key,
                down_payment_pct=down_payment_pct / 100,
                mortgage_rate=mortgage_rate / 100,
                mortgage_term_years=mortgage_term,
                time_horizon_years=time_horizon,
            )
        else:
            results = run_custom(
                home_price=home_price,
                monthly_rent=monthly_rent,
                down_payment_pct=down_payment_pct / 100,
                mortgage_rate=mortgage_rate / 100,
                mortgage_term_years=mortgage_term,
                time_horizon_years=time_horizon,
            )

    # ── Key metrics ──
    st.markdown("## Results")
    if selected_city_key != "custom":
        st.markdown(f"**{results['city']}** — ${results['home_price']:,.0f} home vs ${results['monthly_rent']:,.0f}/mo rent")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Buying Wins In", f"{results['buy_wins_pct']:.0f}% of sims")
    with col2:
        be = results["breakeven_year"]
        st.metric("Break-Even Year", f"~{be} years" if be else "Never")
    with col3:
        med = results["median_advantage"]
        winner = "Buying" if med > 0 else "Renting"
        st.metric(f"Median Advantage ({winner})", f"${abs(med):,.0f}")
    with col4:
        st.metric("Downside Risk (10th %ile)", f"${results['p10_advantage']:,.0f}")

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
    # ── Landing page ──
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-number">01</div>
        <div class="step-title">Pick a City or Go Custom</div>
        <div class="step-desc">Choose from 14 cities with real market data, or enter your own numbers.</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-number">02</div>
        <div class="step-title">We Run 10,000 Simulations</div>
        <div class="step-desc">Each one randomizes home prices, rent inflation, and maintenance costs using historical distributions.</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-number">03</div>
        <div class="step-title">See the Probability</div>
        <div class="step-desc">Not one number. A full distribution — so you know the odds, not just the guess.</div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Show available cities as styled cards
    st.markdown("#### Available Cities")
    cities_data = list_cities()
    city_cols = st.columns(4)
    for i, city in enumerate(cities_data):
        from city_data import get_city
        info = get_city(city["key"])
        with city_cols[i % 4]:
            st.markdown(f"""
            <div class="city-card">
                <div class="city-name">{info['name']}</div>
                <div class="city-price">${info['median_home_price']:,.0f} · ${info['median_rent']:,.0f}/mo</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="data-footer">S&P 500 Total Returns · FHFA + NAR + Shiller Home Prices · BLS CPI Rent Inflation</div>', unsafe_allow_html=True)