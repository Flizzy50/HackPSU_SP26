"""
Rent vs Buy — Monte Carlo Simulation Engine
=============================================
Pure-Python (NumPy) backend that runs N simulated futures comparing
two parallel financial lives:
    1. BUY  — purchase a home with a mortgage
    2. RENT — rent + invest the cost difference in the stock market

Every simulation randomizes:
    • Home appreciation rate
    • Rent inflation rate
    • Stock-market returns (for the renter's portfolio)
    • Maintenance costs (as % of home value)

All monetary values are nominal (not inflation-adjusted).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Parameter distributions (historical defaults)
# ──────────────────────────────────────────────

@dataclass
class Distributions:
    """
    Statistical distributions used to randomize each simulation run.
    All values are annualized percentages expressed as decimals
    (e.g., 0.04 = 4 %).

    Defaults are calibrated to broad U.S. historical data:
        • Home appreciation:  ~3.5 % mean  (Case-Shiller long-run)
        • Rent inflation:     ~3.5 % mean  (CPI shelter component)
        • S&P 500 returns:    ~10 % nominal mean, ~16 % stdev
        • Maintenance costs:  ~1.5 % of home value / year
    """
    # Home appreciation  (normal)
    home_appreciation_mean: float = 0.035
    home_appreciation_std:  float = 0.06

    # Rent inflation  (normal)
    rent_inflation_mean: float = 0.035
    rent_inflation_std:  float = 0.02

    # Stock market nominal return  (normal)
    stock_return_mean: float = 0.10
    stock_return_std:  float = 0.16

    # Annual maintenance as fraction of current home value  (normal, clipped ≥ 0)
    maintenance_pct_mean: float = 0.015
    maintenance_pct_std:  float = 0.005


# ──────────────────────────────────────────────
# Simulation inputs
# ──────────────────────────────────────────────

@dataclass
class SimulationInputs:
    home_price:          float          # Purchase price ($)
    down_payment_pct:    float          # e.g. 0.20 for 20 %
    mortgage_rate:       float          # Annual rate, e.g. 0.07
    mortgage_term_years: int   = 30    # Loan term
    monthly_rent:        float = 1500.0 # Starting monthly rent
    time_horizon_years:  int   = 10    # How many years to simulate
    annual_property_tax_rate: float = 0.015   # % of home value / year
    annual_insurance_rate:    float = 0.005   # % of home value / year
    closing_cost_buy_pct:     float = 0.03    # One-time at purchase
    closing_cost_sell_pct:    float = 0.06    # When you sell (agent fees etc.)
    filing_status:       str   = "single"     # For potential tax logic later
    n_simulations:       int   = 10_000
    distributions:       Distributions = field(default_factory=Distributions)


# ──────────────────────────────────────────────
# Simulation results
# ──────────────────────────────────────────────

@dataclass
class SimulationResults:
    # Per-simulation final net wealth
    buy_wealth:  np.ndarray   # shape (n_simulations,)
    rent_wealth: np.ndarray   # shape (n_simulations,)

    # Derived summary stats
    buy_wins_pct:    float
    median_buy:      float
    median_rent:     float
    mean_buy:        float
    mean_rent:       float
    p10_advantage:   float   # 10th-percentile (buy − rent)
    p90_advantage:   float   # 90th-percentile (buy − rent)
    breakeven_year:  Optional[float]  # Earliest year where buy median > rent median

    # Year-by-year trajectories (medians + percentile bands)
    yearly_buy_median:  np.ndarray   # shape (time_horizon,)
    yearly_rent_median: np.ndarray
    yearly_buy_p25:     np.ndarray
    yearly_buy_p75:     np.ndarray
    yearly_rent_p25:    np.ndarray
    yearly_rent_p75:    np.ndarray

    # Sensitivity results  {param_name: correlation_with_advantage}
    sensitivity: dict


# ──────────────────────────────────────────────
# Mortgage math
# ──────────────────────────────────────────────

def monthly_mortgage_payment(principal: float, annual_rate: float, term_years: int) -> float:
    """Fixed-rate mortgage monthly payment (standard amortization formula)."""
    if annual_rate == 0:
        return principal / (term_years * 12)
    r = annual_rate / 12
    n = term_years * 12
    return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


def mortgage_balance_after(principal: float, annual_rate: float,
                           term_years: int, months_elapsed: int) -> float:
    """Remaining balance on a fixed-rate mortgage after `months_elapsed` payments."""
    if annual_rate == 0:
        pmt = principal / (term_years * 12)
        return max(principal - pmt * months_elapsed, 0.0)
    r = annual_rate / 12
    n = term_years * 12
    if months_elapsed >= n:
        return 0.0
    return principal * ((1 + r) ** n - (1 + r) ** months_elapsed) / ((1 + r) ** n - 1)


# ──────────────────────────────────────────────
# Core simulation  (vectorized across simulations)
# ──────────────────────────────────────────────

def run_simulation(inputs: SimulationInputs) -> SimulationResults:
    """
    Run the full Monte Carlo simulation.

    For each of `n_simulations` runs and each year in `time_horizon_years`:
        BUY path:
            - Home value grows by a random appreciation rate
            - Owner pays: mortgage, property tax, insurance, maintenance
            - Equity = home_value − remaining_mortgage − selling_costs
        RENT path:
            - Renter pays monthly rent (grows by random rent inflation)
            - Renter invests (buyer_monthly_cost − rent) each month
              into the stock market at a random annual return
            - Wealth = investment portfolio value

    The "cost difference" can be negative (rent > own costs) in which
    case the buyer would be the one with extra cash to invest.  We handle
    both directions.
    """
    rng = np.random.default_rng()
    N = inputs.n_simulations
    T = inputs.time_horizon_years
    d = inputs.distributions

    # ---- Random draws: shape (N, T) ----
    home_appr   = rng.normal(d.home_appreciation_mean, d.home_appreciation_std, (N, T))
    rent_infl   = rng.normal(d.rent_inflation_mean,    d.rent_inflation_std,    (N, T))
    stock_ret   = rng.normal(d.stock_return_mean,       d.stock_return_std,      (N, T))
    maint_pct   = np.clip(
        rng.normal(d.maintenance_pct_mean, d.maintenance_pct_std, (N, T)),
        0.005, None,
    )

    # ---- Fixed quantities ----
    down_payment = inputs.home_price * inputs.down_payment_pct
    loan_amount  = inputs.home_price - down_payment
    monthly_pmt  = monthly_mortgage_payment(loan_amount, inputs.mortgage_rate, inputs.mortgage_term_years)
    closing_buy  = inputs.home_price * inputs.closing_cost_buy_pct

    # ---- Year-by-year arrays ----
    # Track wealth at end of each year for percentile bands
    buy_wealth_yearly  = np.zeros((N, T))
    rent_wealth_yearly = np.zeros((N, T))

    # State vectors (per simulation)
    home_value     = np.full(N, inputs.home_price, dtype=np.float64)
    rent_portfolio = np.zeros(N, dtype=np.float64)
    buy_portfolio  = np.zeros(N, dtype=np.float64)   # Buyer can also invest surplus
    monthly_rent   = np.full(N, inputs.monthly_rent, dtype=np.float64)

    # Initial cash outlay for buyer: down payment + closing costs
    # The renter keeps that cash and invests it immediately.
    rent_portfolio[:] = down_payment + closing_buy

    for t in range(T):
        # ---- Home appreciation ----
        home_value *= (1 + home_appr[:, t])

        # ---- Annual costs for the BUYER (expressed monthly) ----
        annual_prop_tax  = home_value * inputs.annual_property_tax_rate
        annual_insurance = home_value * inputs.annual_insurance_rate
        annual_maint     = home_value * maint_pct[:, t]

        buyer_monthly_cost = (
            monthly_pmt
            + annual_prop_tax / 12
            + annual_insurance / 12
            + annual_maint / 12
        )

        # ---- Monthly cost difference ----
        #   positive → renter has leftover cash to invest
        #   negative → buyer has leftover cash to invest
        diff = buyer_monthly_cost - monthly_rent

        renter_monthly_invest = np.maximum(diff, 0)
        buyer_monthly_invest  = np.maximum(-diff, 0)

        # ---- Grow portfolios monthly (approximate: use annual return / 12) ----
        monthly_return = stock_ret[:, t] / 12
        for _month in range(12):
            rent_portfolio = rent_portfolio * (1 + monthly_return) + renter_monthly_invest
            buy_portfolio  = buy_portfolio  * (1 + monthly_return) + buyer_monthly_invest

        # ---- Rent inflation for next year ----
        monthly_rent *= (1 + rent_infl[:, t])

        # ---- Snapshot wealth at end of year t ----
        months_elapsed = (t + 1) * 12
        remaining_mortgage = np.array([
            mortgage_balance_after(loan_amount, inputs.mortgage_rate,
                                   inputs.mortgage_term_years, months_elapsed)
        ])  # scalar, same for all sims
        sell_costs = home_value * inputs.closing_cost_sell_pct
        buy_equity = home_value - remaining_mortgage - sell_costs + buy_portfolio

        buy_wealth_yearly[:, t]  = buy_equity.ravel()
        rent_wealth_yearly[:, t] = rent_portfolio

    # ──── Final results ────
    final_buy  = buy_wealth_yearly[:, -1]
    final_rent = rent_wealth_yearly[:, -1]
    advantage  = final_buy - final_rent

    # Breakeven year: first year where median buy > median rent
    buy_medians  = np.median(buy_wealth_yearly, axis=0)
    rent_medians = np.median(rent_wealth_yearly, axis=0)
    breakeven = None
    for yr in range(T):
        if buy_medians[yr] > rent_medians[yr]:
            breakeven = yr + 1  # 1-indexed year
            break

    # ──── Sensitivity analysis ────
    # Correlate each random variable's MEAN across years with the advantage
    sensitivity = {}
    for name, draws in [
        ("home_appreciation", home_appr),
        ("rent_inflation",    rent_infl),
        ("stock_returns",     stock_ret),
        ("maintenance_costs", maint_pct),
    ]:
        avg_draw = draws.mean(axis=1)  # average over years, per sim
        corr = np.corrcoef(avg_draw, advantage)[0, 1]
        sensitivity[name] = round(float(corr), 4)

    return SimulationResults(
        buy_wealth  = final_buy,
        rent_wealth = final_rent,
        buy_wins_pct = float((final_buy > final_rent).mean()) * 100,
        median_buy   = float(np.median(final_buy)),
        median_rent  = float(np.median(final_rent)),
        mean_buy     = float(np.mean(final_buy)),
        mean_rent    = float(np.mean(final_rent)),
        p10_advantage = float(np.percentile(advantage, 10)),
        p90_advantage = float(np.percentile(advantage, 90)),
        breakeven_year = breakeven,
        yearly_buy_median  = buy_medians,
        yearly_rent_median = rent_medians,
        yearly_buy_p25  = np.percentile(buy_wealth_yearly, 25, axis=0),
        yearly_buy_p75  = np.percentile(buy_wealth_yearly, 75, axis=0),
        yearly_rent_p25 = np.percentile(rent_wealth_yearly, 25, axis=0),
        yearly_rent_p75 = np.percentile(rent_wealth_yearly, 75, axis=0),
        sensitivity = sensitivity,
    )