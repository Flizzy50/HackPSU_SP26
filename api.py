"""
API layer between the Streamlit frontend and the simulation engine.
==================================================================
Person B imports this module and calls `run_for_city()` or `run_custom()`.
Returns plain dicts that are easy to feed into Plotly charts.
"""

from engine import SimulationInputs, Distributions, run_simulation, SimulationResults
from city_data import get_city, list_cities
import numpy as np


def results_to_dict(results: SimulationResults, inputs: SimulationInputs) -> dict:
    """
    Convert SimulationResults into a JSON-serializable dict
    that the frontend can consume directly.
    """
    years = list(range(1, inputs.time_horizon_years + 1))

    return {
        # Summary
        "buy_wins_pct":    round(results.buy_wins_pct, 1),
        "median_buy":      round(results.median_buy),
        "median_rent":     round(results.median_rent),
        "mean_buy":        round(results.mean_buy),
        "mean_rent":       round(results.mean_rent),
        "median_advantage": round(results.median_buy - results.median_rent),
        "p10_advantage":   round(results.p10_advantage),
        "p90_advantage":   round(results.p90_advantage),
        "breakeven_year":  results.breakeven_year,

        # Trajectories (for wealth-over-time chart)
        "years": years,
        "buy_median_trajectory":  results.yearly_buy_median.tolist(),
        "rent_median_trajectory": results.yearly_rent_median.tolist(),
        "buy_p25_trajectory":     results.yearly_buy_p25.tolist(),
        "buy_p75_trajectory":     results.yearly_buy_p75.tolist(),
        "rent_p25_trajectory":    results.yearly_rent_p25.tolist(),
        "rent_p75_trajectory":    results.yearly_rent_p75.tolist(),

        # Histogram data (raw arrays for Plotly)
        "buy_wealth_distribution":  results.buy_wealth.tolist(),
        "rent_wealth_distribution": results.rent_wealth.tolist(),

        # Sensitivity
        "sensitivity": results.sensitivity,
    }


def run_for_city(
    city_key: str,
    down_payment_pct: float = 0.20,
    mortgage_rate: float = 0.07,
    mortgage_term_years: int = 30,
    time_horizon_years: int = 10,
    n_simulations: int = 10_000,
    distributions: Distributions | None = None,
    renter_savings_rate: float = 1.0,
    renter_keeps_down_payment: bool = True,
    invest_surplus: bool = True,
    buyer_invest_surplus: bool | None = None,
    renter_invest_surplus: bool | None = None,
    monthly_rent_override: float | None = None,
    pmi_rate: float = 0.0,
    hoa_monthly: float = 0.0,
    renter_insurance_monthly: float = 0.0,
) -> dict:
    """
    Run simulation using a preset city's data.
    Returns a dict ready for the frontend.
    """
    city = get_city(city_key)
    dist = distributions or Distributions()

    inputs = SimulationInputs(
        home_price=city["median_home_price"],
        down_payment_pct=down_payment_pct,
        mortgage_rate=mortgage_rate,
        mortgage_term_years=mortgage_term_years,
        monthly_rent=monthly_rent_override if monthly_rent_override is not None else city["median_rent"],
        annual_property_tax_rate=city["property_tax_rate"],
        annual_insurance_rate=city["insurance_rate"],
        time_horizon_years=time_horizon_years,
        n_simulations=n_simulations,
        distributions=dist,
        renter_savings_rate=renter_savings_rate,
        renter_keeps_down_payment=renter_keeps_down_payment,
        invest_surplus=invest_surplus,
        buyer_invest_surplus=buyer_invest_surplus if buyer_invest_surplus is not None else invest_surplus,
        renter_invest_surplus=renter_invest_surplus if renter_invest_surplus is not None else invest_surplus,
        pmi_rate=pmi_rate,
        hoa_monthly=hoa_monthly,
        renter_insurance_monthly=renter_insurance_monthly,
    )

    results = run_simulation(inputs)
    output = results_to_dict(results, inputs)
    output["city"] = city["name"]
    output["home_price"] = city["median_home_price"]
    output["monthly_rent"] = inputs.monthly_rent
    return output


def run_custom(
    home_price: float,
    monthly_rent: float,
    down_payment_pct: float = 0.20,
    mortgage_rate: float = 0.07,
    mortgage_term_years: int = 30,
    time_horizon_years: int = 10,
    property_tax_rate: float = 0.015,
    insurance_rate: float = 0.005,
    n_simulations: int = 10_000,
    distributions: Distributions | None = None,
    renter_savings_rate: float = 1.0,
    renter_keeps_down_payment: bool = True,
    invest_surplus: bool = False,
    buyer_invest_surplus: bool | None = None,
    renter_invest_surplus: bool | None = None,
    pmi_rate: float = 0.0,
    hoa_monthly: float = 0.0,
    renter_insurance_monthly: float = 0.0,
) -> dict:
    """
    Run simulation with fully custom inputs (no city preset).
    Returns a dict ready for the frontend.
    """
    dist = distributions or Distributions()

    inputs = SimulationInputs(
        home_price=home_price,
        down_payment_pct=down_payment_pct,
        mortgage_rate=mortgage_rate,
        mortgage_term_years=mortgage_term_years,
        monthly_rent=monthly_rent,
        annual_property_tax_rate=property_tax_rate,
        annual_insurance_rate=insurance_rate,
        time_horizon_years=time_horizon_years,
        n_simulations=n_simulations,
        distributions=dist,
        renter_savings_rate=renter_savings_rate,
        renter_keeps_down_payment=renter_keeps_down_payment,
        invest_surplus=invest_surplus,
        buyer_invest_surplus=buyer_invest_surplus if buyer_invest_surplus is not None else invest_surplus,
        renter_invest_surplus=renter_invest_surplus if renter_invest_surplus is not None else invest_surplus,
        pmi_rate=pmi_rate,
        hoa_monthly=hoa_monthly,
        renter_insurance_monthly=renter_insurance_monthly,
    )

    results = run_simulation(inputs)
    output = results_to_dict(results, inputs)
    output["city"] = "Custom"
    output["home_price"] = home_price
    output["monthly_rent"] = monthly_rent
    return output