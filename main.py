"""
Revenue Forecasting — Main Entry Point
=======================================
Run this file to:
  1. Generate (or load) revenue data
  2. Train the Linear Regression model
  3. Evaluate on test set
  4. Predict next month's revenue
  5. Save plots and model

Usage:
  python main.py                        # Generate synthetic data + run full pipeline
  python main.py --data your_data.csv   # Use your own CSV (needs 'date', 'revenue' columns)
  python main.py --model ridge          # Use Ridge Regression instead
"""

import argparse
import os
import sys

# Make sure imports work regardless of where script is run from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_generator import generate_revenue_data
from models.revenue_model import RevenueForecastingModel


def main():
    parser = argparse.ArgumentParser(description="Revenue Forecasting with Linear Regression")
    parser.add_argument("--data",  type=str, default=None,     help="Path to CSV file (optional)")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "ridge"],
                        help="Model type: linear or ridge (default: linear)")
    args = parser.parse_args()

    print("=" * 60)
    print("   REVENUE FORECASTING MODEL — Pavithra Nagineni")
    print("=" * 60)


    data_path = args.data if args.data else "data/revenue_data.csv"

    if not args.data:
        print("\n📂 No data file provided. Generating synthetic data...")
        generate_revenue_data(months=36, save_path=data_path)
    else:
        print(f"\n📂 Using provided data: {data_path}")


    print(f"\n🤖 Model type: {args.model.upper()} REGRESSION")
    model = RevenueForecastingModel(model_type=args.model)


    print("\n📥 Loading and preprocessing data...")
    df = model.load_data(data_path)
    df = model.preprocess(df)


    print("\n🏋️  Training model...")
    metrics = model.train(df)

    model.feature_importance()


    print("\n🔮 Forecasting next month...")
    next_revenue = model.predict_next_month()


    print("\n📊 Generating forecast dashboard...")
    os.makedirs("outputs", exist_ok=True)
    model.visualize(save_path="outputs/forecast_plot.png")


    model.save_model(path="outputs/model.pkl")


    print("\n" + "=" * 60)
    print("   FORECAST SUMMARY")
    print("=" * 60)
    print(f"   Model Type      : {args.model.upper()} REGRESSION")
    print(f"   MAE             : ${metrics['MAE']:,.2f}")
    print(f"   RMSE            : ${metrics['RMSE']:,.2f}")
    print(f"   R² Score        : {metrics['R2']:.4f}")
    print(f"   Next Month Rev  : ${next_revenue:,.2f}")
    print(f"   Plot saved to   : outputs/forecast_plot.png")
    print(f"   Model saved to  : outputs/model.pkl")
    print("=" * 60)
    print("\n✅ Done! Check the 'outputs/' folder for results.\n")


if __name__ == "__main__":
    main()
