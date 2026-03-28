"""
Tests for the Monte Carlo engine.
Run with:  python -m pytest tests.py -v
Or just:   python tests.py
"""

import numpy as np
import sys
from engine import (
    monthly_mortgage_payment,
    mortgage_balance_after,
    run_simulation,
    SimulationInputs,
    Distributions,
)
from city_data import get_city, list_cities


# ──────────────────────────────────────────────
# Mortgage math tests
# ──────────────────────────────────────────────

def test_monthly_payment_known_value():
    """$300k loan, 7%, 30yr → should be ~$1,995.91"""
    pmt = monthly_mortgage_payment(300_000, 0.07, 30)
    assert abs(pmt - 1995.91) < 1.0, f"Expected ~1995.91, got {pmt:.2f}"
    print(f"  [OK] Monthly payment: ${pmt:,.2f}")


def test_monthly_payment_zero_rate():
    """0% interest → simple division."""
    pmt = monthly_mortgage_payment(360_000, 0.0, 30)
    assert abs(pmt - 1000.0) < 0.01, f"Expected 1000.0, got {pmt:.2f}"
    print(f"  [OK] Zero-rate payment: ${pmt:,.2f}")


def test_balance_after_full_term():
    """Balance should be ~0 after full term."""
    bal = mortgage_balance_after(300_000, 0.07, 30, 360)
    assert abs(bal) < 1.0, f"Expected ~0, got {bal:.2f}"
    print(f"  [OK] Balance after 360 months: ${bal:,.2f}")


def test_balance_after_zero_months():
    """Balance at month 0 = full principal."""
    bal = mortgage_balance_after(300_000, 0.07, 30, 0)
    assert abs(bal - 300_000) < 1.0, f"Expected 300000, got {bal:.2f}"
    print(f"  [OK] Balance at month 0: ${bal:,.2f}")


def test_balance_decreases_over_time():
    """Balance should decrease monotonically."""
    balances = [mortgage_balance_after(300_000, 0.07, 30, m) for m in range(0, 361, 12)]
    for i in range(1, len(balances)):
        assert balances[i] < balances[i - 1], f"Balance did not decrease at year {i}"
    print(f"  [OK] Balance decreases: ${balances[0]:,.0f} -> ${balances[-1]:,.0f}")


# ──────────────────────────────────────────────
# Simulation smoke tests
# ──────────────────────────────────────────────

def test_simulation_runs():
    """Smoke test: simulation completes and returns valid shapes."""
    inputs = SimulationInputs(
        home_price=340_000,
        down_payment_pct=0.20,
        mortgage_rate=0.07,
        monthly_rent=1_500,
        time_horizon_years=10,
        n_simulations=1_000,  # fewer for speed in tests
    )
    results = run_simulation(inputs)

    assert results.buy_wealth.shape == (1_000,)
    assert results.rent_wealth.shape == (1_000,)
    assert 0 <= results.buy_wins_pct <= 100
    assert results.yearly_buy_median.shape == (10,)
    assert len(results.sensitivity) == 3
    print(f"  [OK] Simulation ran: buy wins {results.buy_wins_pct:.1f}% of the time")
    print(f"    Median buy wealth:  ${results.median_buy:,.0f}")
    print(f"    Median rent wealth: ${results.median_rent:,.0f}")
    print(f"    Breakeven year:     {results.breakeven_year}")
    print(f"    Sensitivity:        {results.sensitivity}")


def test_simulation_short_horizon():
    """Short horizon (2 years): buying almost always loses due to transaction costs."""
    inputs = SimulationInputs(
        home_price=340_000,
        down_payment_pct=0.20,
        mortgage_rate=0.07,
        monthly_rent=1_500,
        time_horizon_years=2,
        n_simulations=2_000,
    )
    results = run_simulation(inputs)
    # With 3% buy + 6% sell closing costs, buying in 2 years should usually lose
    assert results.buy_wins_pct < 50, (
        f"Expected buy to lose short-term, but it won {results.buy_wins_pct:.1f}%"
    )
    print(f"  [OK] Short horizon: buy wins only {results.buy_wins_pct:.1f}% (expected < 50%)")


def test_simulation_long_horizon():
    """Long horizon (30 years) with cheap house and expensive rent: buying should usually win."""
    inputs = SimulationInputs(
        home_price=200_000,
        down_payment_pct=0.20,
        mortgage_rate=0.05,
        monthly_rent=1_800,  # rent >> mortgage
        time_horizon_years=30,
        n_simulations=2_000,
    )
    results = run_simulation(inputs)
    assert results.buy_wins_pct > 50, (
        f"Expected buy to win long-term, but only {results.buy_wins_pct:.1f}%"
    )
    print(f"  [OK] Long horizon: buy wins {results.buy_wins_pct:.1f}% (expected > 50%)")


def test_sensitivity_has_expected_keys():
    """Sensitivity dict should have all 4 factors."""
    inputs = SimulationInputs(
        home_price=340_000,
        down_payment_pct=0.20,
        mortgage_rate=0.07,
        monthly_rent=1_500,
        time_horizon_years=10,
        n_simulations=500,
    )
    results = run_simulation(inputs)
    expected_keys = {"home_appreciation", "rent_inflation", "maintenance_costs"}
    assert set(results.sensitivity.keys()) == expected_keys
    print(f"  [OK] Sensitivity keys present: {list(results.sensitivity.keys())}")


# ──────────────────────────────────────────────
# City data tests
# ──────────────────────────────────────────────

def test_city_data():
    """Verify city data loads and State College exists."""
    cities = list_cities()
    assert len(cities) >= 5
    sc = get_city("state_college_pa")
    assert sc["median_home_price"] > 0
    assert sc["median_rent"] > 0
    print(f"  [OK] {len(cities)} cities loaded. State College: ${sc['median_home_price']:,} / ${sc['median_rent']:,}/mo")


# ──────────────────────────────────────────────
# Integration test: city data → engine
# ──────────────────────────────────────────────

def test_city_to_engine():
    """Wire city data into the engine and run."""
    city = get_city("state_college_pa")
    inputs = SimulationInputs(
        home_price=city["median_home_price"],
        down_payment_pct=0.20,
        mortgage_rate=0.07,
        monthly_rent=city["median_rent"],
        annual_property_tax_rate=city["property_tax_rate"],
        annual_insurance_rate=city["insurance_rate"],
        time_horizon_years=10,
        n_simulations=1_000,
    )
    results = run_simulation(inputs)
    assert results.buy_wealth.shape == (1_000,)
    print(f"  [OK] State College sim: buy wins {results.buy_wins_pct:.1f}%")
    print(f"    Median advantage: ${results.median_buy - results.median_rent:,.0f}")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_monthly_payment_known_value,
        test_monthly_payment_zero_rate,
        test_balance_after_full_term,
        test_balance_after_zero_months,
        test_balance_decreases_over_time,
        test_simulation_runs,
        test_simulation_short_horizon,
        test_simulation_long_horizon,
        test_sensitivity_has_expected_keys,
        test_city_data,
        test_city_to_engine,
    ]
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            print(f"\n{name}:")
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
