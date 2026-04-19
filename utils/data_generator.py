"""
Generate realistic synthetic revenue data for the forecasting model.
Simulates 3 years of monthly revenue with:
  - Upward trend
  - Seasonality (Q4 spike)
  - Noise
  - Invoice count & customer count as extra features
"""

import pandas as pd
import numpy as np
import os


def generate_revenue_data(months: int = 36, save_path: str = "data/revenue_data.csv") -> pd.DataFrame:
    """
    Generate synthetic monthly revenue data.

    Columns:
      date         - First day of each month
      revenue      - Monthly revenue in USD
      invoices     - Number of invoices processed
      customers    - Number of active customers
      avg_deal_size- Average deal size per invoice
    """
    np.random.seed(42)

    start_date = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start=start_date, periods=months, freq="MS")

    # Base revenue with upward trend
    base_revenue = 100_000
    trend        = np.linspace(0, 80_000, months)           # grows $80k over 3 years

    # Seasonality: Q4 spike, Q1 dip
    month_effects = {
        1: -0.10, 2: -0.05, 3:  0.02,  # Q1 — post-holiday dip
        4:  0.03, 5:  0.05, 6:  0.08,  # Q2 — moderate
        7:  0.04, 8:  0.03, 9:  0.06,  # Q3 — stable
        10: 0.12, 11: 0.18, 12: 0.22   # Q4 — year-end push
    }
    seasonality = np.array([month_effects[d.month] for d in dates])

    # Random noise
    noise = np.random.normal(0, 5000, months)

    revenue = base_revenue + trend + (base_revenue * seasonality) + noise
    revenue = np.maximum(revenue, 50_000)  # Floor at $50k

    # Correlated features
    invoices      = (revenue / 420 + np.random.normal(0, 5, months)).astype(int)
    customers     = (revenue / 2800 + np.random.normal(0, 2, months)).astype(int)
    avg_deal_size = revenue / invoices

    df = pd.DataFrame({
        "date"         : dates,
        "revenue"      : revenue.round(2),
        "invoices"     : invoices,
        "customers"    : customers,
        "avg_deal_size": avg_deal_size.round(2),
    })

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Generated {months} months of data → saved to {save_path}")
    print(f"   Revenue range: ${df['revenue'].min():,.0f} – ${df['revenue'].max():,.0f}")
    return df


if __name__ == "__main__":
    df = generate_revenue_data()
    print(df.tail())
