"""
Rent vs Buy — Monte Carlo Simulation Engine
=============================================
Pure-Python (NumPy) backend that runs N simulated futures comparing
two parallel financial lives:
    1. BUY  — purchase a home with a mortgage
    2. RENT — rent and keep part of the cost difference as cash (no investing)

Every simulation randomizes:
    • Home appreciation rate
    • Rent inflation rate
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
        • Maintenance costs:  ~1.5 % of home value / year
    """
    # Home appreciation  (normal)
    home_appreciation_mean: float = 0.035
    home_appreciation_std:  float = 0.06

    # Rent inflation  (normal)
    rent_inflation_mean: float = 0.035
    rent_inflation_std:  float = 0.02

    # Stock market return fields retained for compatibility (unused when investing is disabled)
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
    renter_savings_rate: float = 1.0          # Fraction of surplus rent the renter keeps (default: all)
    renter_keeps_down_payment: bool = True    # If True, renter starts with buyer's DP + closing cash
    invest_surplus:      bool  = False        # Legacy toggle: when True, both sides invest surplus
    buyer_invest_surplus: bool = False        # Buyer invests surplus cash at stock_return assumptions
    renter_invest_surplus: bool = False       # Renter invests surplus cash at stock_return assumptions
    # Additional cash-flow factors commonly used by rent vs buy calculators
    pmi_rate:            float = 0.0          # Annual PMI rate on outstanding balance when LTV >= 80%
    hoa_monthly:         float = 0.0          # HOA / common charges ($/mo)
    utilities_delta:     float = 0.0          # Extra monthly utilities owners pay vs renters ($/mo)
    renter_insurance_monthly: float = 0.0     # Renter's insurance ($/mo)
    renter_security_deposit_months: float = 1.0  # Months of rent held as deposit (returned at end)
    renter_broker_fee_pct: float = 0.0        # Percent of annual rent paid upfront to broker


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
        Rent path:
            - Renter pays monthly rent (grows by random rent inflation)
            - Renter keeps only a portion of any monthly cost advantage as idle cash
              (e.g., they spend most of the surplus)

        The "cost difference" can be negative (rent > own costs) in which
        case the buyer keeps the surplus cash.  No investing/compounding is applied.
    """
    rng = np.random.default_rng()
    N = inputs.n_simulations
    T = inputs.time_horizon_years
    d = inputs.distributions

    # ---- Random draws: shape (N, T) ----
    home_appr   = rng.normal(d.home_appreciation_mean, d.home_appreciation_std, (N, T))
    rent_infl   = rng.normal(d.rent_inflation_mean,    d.rent_inflation_std,    (N, T))
    maint_pct   = np.clip(
        rng.normal(d.maintenance_pct_mean, d.maintenance_pct_std, (N, T)),
        0.0, None,
    )
    stock_ret   = rng.normal(d.stock_return_mean, d.stock_return_std, (N, T))

    # ---- Fixed quantities ----
    down_payment = inputs.home_price * inputs.down_payment_pct
    loan_amount  = inputs.home_price - down_payment
    monthly_pmt  = monthly_mortgage_payment(loan_amount, inputs.mortgage_rate, inputs.mortgage_term_years)
    term_months  = inputs.mortgage_term_years * 12
    closing_buy  = inputs.home_price * inputs.closing_cost_buy_pct
    # Security deposit (returned at end) and broker fee (not returned)
    renter_deposit = inputs.monthly_rent * inputs.renter_security_deposit_months
    renter_broker_fee = inputs.renter_broker_fee_pct * inputs.monthly_rent * 12

    # ---- Year-by-year arrays ----
    # Track wealth at end of each year for percentile bands
    buy_wealth_yearly  = np.zeros((N, T))
    rent_wealth_yearly = np.zeros((N, T))

    # State vectors (per simulation)
    home_value     = np.full(N, inputs.home_price, dtype=np.float64)
    rent_portfolio = np.zeros(N, dtype=np.float64)  # renter surplus (invested or cash)
    buy_portfolio  = np.zeros(N, dtype=np.float64)  # buyer surplus (invested or cash)
    monthly_rent   = np.full(N, inputs.monthly_rent, dtype=np.float64)

    # Initial cash outlay for buyer: down payment + closing costs
    # Upfront flows
    # Buyer: pays closing costs immediately.
    buy_portfolio[:] -= closing_buy
    # Renter parity: always start renter with the buyer's upfront cash
    # so the comparison isn't driven by a single flag.
    rent_portfolio[:] = down_payment + closing_buy
    rent_portfolio[:] -= renter_deposit + renter_broker_fee

    for t in range(T):
        # ---- Home appreciation ----
        home_value *= np.clip((1 + home_appr[:, t]), 0.0, None)

        # ---- Annual costs for the BUYER (expressed monthly) ----
        annual_prop_tax  = home_value * inputs.annual_property_tax_rate
        annual_insurance = home_value * inputs.annual_insurance_rate
        annual_maint     = home_value * maint_pct[:, t]

        # Private mortgage insurance until LTV < 80%
        remaining_balance_for_pmi = loan_amount  # updated below per month via balance helper
        pmi_monthly = 0.0
        if inputs.pmi_rate > 0 and inputs.down_payment_pct < 0.20:
            # Approximated each year using current LTV; recomputed monthly inside loop.
            pass

        base_owner_cost = (
            annual_prop_tax / 12
            + annual_insurance / 12
            + annual_maint / 12
            + inputs.hoa_monthly
            + inputs.utilities_delta
        )

        # ---- Grow/save portfolios monthly ----
        buyer_invest_flag = inputs.buyer_invest_surplus or inputs.invest_surplus
        renter_invest_flag = inputs.renter_invest_surplus or inputs.invest_surplus
        monthly_return_buyer = (stock_ret[:, t] / 12) if buyer_invest_flag else 0.0
        monthly_return_renter = (stock_ret[:, t] / 12) if renter_invest_flag else 0.0
        for _month in range(12):
            months_elapsed = t * 12 + (_month + 1)
            # PMI applied monthly until LTV < 80%
            if inputs.pmi_rate > 0 and inputs.down_payment_pct < 0.20:
                remaining_balance_for_pmi = mortgage_balance_after(
                    loan_amount, inputs.mortgage_rate, inputs.mortgage_term_years, months_elapsed
                )
                ltv = np.divide(remaining_balance_for_pmi, home_value, out=np.zeros_like(home_value), where=home_value > 0)
                pmi_monthly = np.where(ltv >= 0.80, remaining_balance_for_pmi * inputs.pmi_rate / 12, 0.0)
            else:
                pmi_monthly = 0.0

            mortgage_component = 0.0 if months_elapsed > term_months else monthly_pmt
            adjusted_buyer_cost = base_owner_cost + mortgage_component + pmi_monthly
            adjusted_rent_cost  = monthly_rent + inputs.renter_insurance_monthly

            diff = adjusted_buyer_cost - adjusted_rent_cost
            renter_monthly_save = np.maximum(diff, 0) * inputs.renter_savings_rate
            buyer_monthly_save  = np.maximum(-diff, 0)

            rent_portfolio = rent_portfolio * (1 + monthly_return_renter) + renter_monthly_save
            buy_portfolio  = buy_portfolio  * (1 + monthly_return_buyer) + buyer_monthly_save

            # After the mortgage is paid off, redirect the freed-up payment into investments
            if months_elapsed > term_months:
                buy_portfolio += monthly_pmt

        # ---- Rent inflation for next year ----
        monthly_rent *= np.clip((1 + rent_infl[:, t]), 0.0, None)

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

    # Add security deposit back at the end of the horizon
    rent_wealth_yearly[:, -1] += renter_deposit

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
        ("maintenance_costs", maint_pct),
        ("stock_returns",     stock_ret),
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