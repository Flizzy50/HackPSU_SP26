"""
BUILD VERIFIED HISTORICAL DATA CSV
===================================
Sources:
  - S&P 500 Total Returns: slickcharts.com (verified against S&P data)
  - Home Appreciation: computed from DQYDJ median home prices (NAR + FHFA + Shiller)
  - Rent Inflation: BLS CPI Rent of Primary Residence via FRED

To get rent data from FRED, you need a free API key:
  1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
  2. Create account, get key (takes 2 min)
  3. Set it below in FRED_API_KEY

If you don't want to bother with FRED, the script will still generate the CSV
using the S&P and home price data, and use reasonable BLS-sourced rent inflation
estimates for the rent column (clearly marked).
"""

import csv
import os

# ── S&P 500 TOTAL RETURNS (VERIFIED from slickcharts.com) ──
# These are total returns including dividends
SP500_RETURNS = {
    1975: 0.3720, 1976: 0.2384, 1977: -0.0718, 1978: 0.0656, 1979: 0.1844,
    1980: 0.3242, 1981: -0.0491, 1982: 0.2155, 1983: 0.2256, 1984: 0.0627,
    1985: 0.3173, 1986: 0.1867, 1987: 0.0525, 1988: 0.1661, 1989: 0.3169,
    1990: -0.0310, 1991: 0.3047, 1992: 0.0762, 1993: 0.1008, 1994: 0.0132,
    1995: 0.3758, 1996: 0.2296, 1997: 0.3336, 1998: 0.2858, 1999: 0.2104,
    2000: -0.0910, 2001: -0.1189, 2002: -0.2210, 2003: 0.2868, 2004: 0.1088,
    2005: 0.0491, 2006: 0.1579, 2007: 0.0549, 2008: -0.3700, 2009: 0.2646,
    2010: 0.1506, 2011: 0.0211, 2012: 0.1600, 2013: 0.3239, 2014: 0.1369,
    2015: 0.0138, 2016: 0.1196, 2017: 0.2183, 2018: -0.0438, 2019: 0.3149,
    2020: 0.1840, 2021: 0.2871, 2022: -0.1811, 2023: 0.2629, 2024: 0.2502,
    2025: 0.1788,
}

# ── MEDIAN HOME PRICES (from DQYDJ.com, sourced from NAR + FHFA + Shiller) ──
# Using January values each year to compute year-over-year appreciation
# These are nominal, non-seasonally-adjusted median existing home sale prices
MEDIAN_HOME_PRICES_JAN = {
    1975: 32033.50, 1976: 33693.12, 1977: 36606.64, 1978: 42044.45,
    1979: 48813.50, 1980: 55074.15, 1981: 59034.91, 1982: 61890.11,
    1983: 62408.66, 1984: 65341.20, 1985: 68432.22, 1986: 73735.64,
    1987: 80829.26, 1988: 86968.19, 1989: 93342.94, 1990: 97024.78,
    1991: 96112.70, 1992: 98207.96, 1993: 100139.82, 1994: 103494.16,
    1995: 106050.75, 1996: 109328.20, 1997: 112028.96, 1998: 116075.31,
    1999: 123158.81, 2000: 131414.89, 2001: 140007.37, 2002: 149147.69,
    2003: 160681.21, 2004: 173368.09, 2005: 191033.60, 2006: 209564.13,
    2007: 213629.70, 2008: 203710.87, 2009: 185295.67, 2010: 180115.20,
    2011: 171686.12, 2012: 169081.46, 2013: 179634.64, 2014: 191100.88,
    2015: 199751.02, 2016: 211169.21, 2017: 223221.75, 2018: 238705.50,
    2019: 251123.26, 2020: 266097.62, 2021: 298920.11, 2022: 352214.60,
    2023: 369784.00, 2024: 393860.23, 2025: 414495.63,
}

# Compute year-over-year home appreciation
HOME_APPRECIATION = {}
years = sorted(MEDIAN_HOME_PRICES_JAN.keys())
for i in range(1, len(years)):
    prev_year = years[i - 1]
    curr_year = years[i]
    appreciation = (MEDIAN_HOME_PRICES_JAN[curr_year] - MEDIAN_HOME_PRICES_JAN[prev_year]) / MEDIAN_HOME_PRICES_JAN[prev_year]
    HOME_APPRECIATION[curr_year] = round(appreciation, 4)

# ── RENT INFLATION (BLS CPI Rent of Primary Residence, annual % change) ──
# Attempt to pull from FRED API if key is available.
# Fallback: use published BLS annual figures.

FRED_API_KEY = ""  # <-- PASTE YOUR FRED API KEY HERE IF YOU HAVE ONE

def get_rent_inflation_from_fred():
    """Try to pull CPI Rent data from FRED API."""
    try:
        import requests
        if not FRED_API_KEY:
            return None

        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id=CUUR0000SEHA&api_key={FRED_API_KEY}"
            f"&file_type=json&observation_start=1974-01-01&frequency=a"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        observations = data.get("observations", [])

        # Compute year-over-year % change from annual index values
        rent_inflation = {}
        prev_val = None
        for obs in observations:
            year = int(obs["date"][:4])
            val = float(obs["value"])
            if prev_val is not None and prev_val > 0:
                rent_inflation[year] = round((val - prev_val) / prev_val, 4)
            prev_val = val

        if len(rent_inflation) > 10:
            print(f"[OK] Pulled {len(rent_inflation)} years of rent CPI data from FRED")
            return rent_inflation
        return None
    except Exception as e:
        print(f"[WARN] Could not fetch from FRED: {e}")
        return None


# BLS-published annual rent CPI percent changes (fallback)
# Source: BLS CPI Detailed Report tables, various years
# These are approximate annual averages for CPI Rent of Primary Residence
RENT_INFLATION_BLS = {
    1976: 0.0500, 1977: 0.0620, 1978: 0.0720, 1979: 0.0810,
    1980: 0.0910, 1981: 0.0990, 1982: 0.0740, 1983: 0.0520,
    1984: 0.0490, 1985: 0.0520, 1986: 0.0460, 1987: 0.0410,
    1988: 0.0370, 1989: 0.0380, 1990: 0.0380, 1991: 0.0370,
    1992: 0.0290, 1993: 0.0260, 1994: 0.0260, 1995: 0.0280,
    1996: 0.0320, 1997: 0.0310, 1998: 0.0310, 1999: 0.0290,
    2000: 0.0360, 2001: 0.0420, 2002: 0.0370, 2003: 0.0280,
    2004: 0.0260, 2005: 0.0290, 2006: 0.0420, 2007: 0.0430,
    2008: 0.0360, 2009: 0.0180, 2010: 0.0060, 2011: 0.0190,
    2012: 0.0250, 2013: 0.0290, 2014: 0.0300, 2015: 0.0360,
    2016: 0.0380, 2017: 0.0370, 2018: 0.0350, 2019: 0.0360,
    2020: 0.0200, 2021: 0.0180, 2022: 0.0590, 2023: 0.0830,
    2024: 0.0500, 2025: 0.0400,
}


def build_csv(output_path="historical_data.csv"):
    # Try FRED first for rent data
    rent_data = get_rent_inflation_from_fred()
    rent_source = "FRED API (BLS CPI CUUR0000SEHA)"

    if rent_data is None:
        rent_data = RENT_INFLATION_BLS
        rent_source = "BLS CPI published annual reports (manually compiled)"
        print("[INFO] Using BLS-published rent inflation fallback")

    # Build combined dataset
    start_year = 1976  # first year we have all three data points
    end_year = min(max(SP500_RETURNS.keys()), max(HOME_APPRECIATION.keys()))

    rows = []
    for year in range(start_year, end_year + 1):
        sp500 = SP500_RETURNS.get(year)
        home = HOME_APPRECIATION.get(year)
        rent = rent_data.get(year)

        if sp500 is not None and home is not None and rent is not None:
            rows.append({
                "year": year,
                "sp500_return": sp500,
                "home_appreciation": home,
                "rent_inflation": rent,
            })

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["year", "sp500_return", "home_appreciation", "rent_inflation"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[DONE] Wrote {len(rows)} rows to {output_path}")
    print(f"\nData sources:")
    print(f"  S&P 500 Returns: slickcharts.com (total returns incl. dividends)")
    print(f"  Home Appreciation: computed YoY from DQYDJ median prices (NAR + FHFA + Shiller)")
    print(f"  Rent Inflation: {rent_source}")
    print(f"\nYears covered: {rows[0]['year']} - {rows[-1]['year']}")

    # Print a few rows for verification
    print(f"\nSample rows for verification:")
    print(f"{'Year':<6} {'S&P 500':<10} {'Home Appr.':<12} {'Rent Infl.':<10}")
    print("-" * 40)
    for row in rows[:5]:
        print(f"{row['year']:<6} {row['sp500_return']:>8.2%}   {row['home_appreciation']:>8.2%}     {row['rent_inflation']:>8.2%}")
    print("...")
    for row in rows[-5:]:
        print(f"{row['year']:<6} {row['sp500_return']:>8.2%}   {row['home_appreciation']:>8.2%}     {row['rent_inflation']:>8.2%}")


if __name__ == "__main__":
    build_csv()
