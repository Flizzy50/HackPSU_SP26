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

st.title("RentOrOwn")
st.markdown("**Every rent vs. buy calculator gives you one number. We give you 10,000.**")
st.markdown("---")

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
        marker_color="#2ecc71", opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=rent_wins, nbinsx=60, name="Renting Wins",
        marker_color="#e74c3c", opacity=0.85,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=2)
    fig.update_layout(
        title="Distribution of Outcomes (10,000 Simulations)",
        xaxis_title="Net Wealth Difference (Buy − Rent)",
        yaxis_title="Number of Simulations",
        barmode="overlay", template="plotly_dark", height=420,
        legend=dict(x=0.02, y=0.98),
        xaxis=dict(tickformat="$,.0f"),
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
        fill="toself", fillcolor="rgba(46, 204, 113, 0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Rent confidence band
    fig.add_trace(go.Scatter(
        x=years + years[::-1],
        y=results["rent_p75_trajectory"] + results["rent_p25_trajectory"][::-1],
        fill="toself", fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False,
    ))
    # Median lines
    fig.add_trace(go.Scatter(
        x=years, y=results["buy_median_trajectory"],
        mode="lines", name="Buy (Median)",
        line=dict(color="#2ecc71", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=years, y=results["rent_median_trajectory"],
        mode="lines", name="Rent + Invest (Median)",
        line=dict(color="#e74c3c", width=3),
    ))

    fig.update_layout(
        title="Wealth Accumulation Over Time",
        xaxis_title="Year", yaxis_title="Net Wealth ($)",
        template="plotly_dark", height=420,
        legend=dict(x=0.02, y=0.98),
        yaxis=dict(tickformat="$,.0f"),
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
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="What Matters Most? (Correlation with Outcome)",
        xaxis_title="Correlation with Buy vs Rent Advantage",
        template="plotly_dark", height=400,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
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
            st.markdown(summary)
        else:
            # Fallback
            pct = results["buy_wins_pct"]
            if pct > 60:
                st.markdown(f"Buying looks favorable — it wins in **{pct:.0f}%** of simulations.")
            elif pct > 40:
                st.markdown(f"It's close to a coin flip. Buying wins **{pct:.0f}%** of the time.")
            else:
                st.markdown(f"Renting and investing looks stronger. Buying only wins **{pct:.0f}%** of the time.")

            be = results["breakeven_year"]
            if be:
                st.markdown(f"Buying breaks even around **year {be}**.")
            else:
                st.markdown("Buying doesn't break even in your time horizon.")

        st.markdown("---")
        st.caption("AI summary powered by Google Gemini. This is a simulation, not financial advice. Built on 50 years of real data from FHFA, BLS, and S&P 500.")

else:
    # ── Landing page ──
    st.markdown("## How It Works")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1. Pick a City or Go Custom")
        st.markdown("Choose from 8 college towns with real data, or enter your own numbers.")
    with col2:
        st.markdown("### 2. We Run 10,000 Simulations")
        st.markdown("Each one randomizes home prices, rent inflation, market returns, and maintenance costs.")
    with col3:
        st.markdown("### 3. See the Probability")
        st.markdown("Not one number. A full distribution — so you know the odds, not just the guess.")

    st.markdown("---")

    # Show available cities
    st.markdown("#### Available Cities")
    cities_data = list_cities()
    city_cols = st.columns(4)
    for i, city in enumerate(cities_data):
        from city_data import get_city
        info = get_city(city["key"])
        with city_cols[i % 4]:
            st.markdown(f"**{info['name']}**")
            st.markdown(f"${info['median_home_price']:,.0f} / ${info['median_rent']:,.0f}/mo")

    st.markdown("---")
    st.markdown("**Data Sources:** S&P 500 total returns · FHFA + NAR + Shiller home prices · BLS CPI rent inflation")