"""
Revenue Forecasting Model using Linear Regression & Time Series
Author: Pavithra Nagineni
Description: Predicts next month's revenue using historical billing data,
             Linear Regression, and Time-Series feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the date column."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)  # Cyclical encoding
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_q4"] = (df["quarter"] == 4).astype(int)          # Q4 seasonality flag
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = "revenue", lags: list = [1, 2, 3]) -> pd.DataFrame:
    """Create lag features — revenue from previous N months."""
    df = df.copy()
    for lag in lags:
        df[f"revenue_lag_{lag}"] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = "revenue") -> pd.DataFrame:
    """Rolling statistics — 3-month and 6-month averages."""
    df = df.copy()
    df["rolling_mean_3"] = df[target_col].shift(1).rolling(window=3).mean()
    df["rolling_mean_6"] = df[target_col].shift(1).rolling(window=6).mean()
    df["rolling_std_3"]  = df[target_col].shift(1).rolling(window=3).std()
    df["revenue_growth"] = df[target_col].pct_change()         # Month-over-month growth %
    return df


# ─────────────────────────────────────────
# 2. MODEL CLASS
# ─────────────────────────────────────────
class RevenueForecastingModel:
    """
    End-to-end Revenue Forecasting Pipeline.

    Steps:
      1. Load & validate data
      2. Feature engineering (time + lag + rolling)
      3. Train Linear Regression (+ Ridge for comparison)
      4. Evaluate: MAE, RMSE, R²
      5. Predict next month's revenue
      6. Visualize actual vs predicted
    """

    def __init__(self, model_type: str = "linear"):
        self.model_type  = model_type
        self.model       = LinearRegression() if model_type == "linear" else Ridge(alpha=1.0)
        self.scaler      = StandardScaler()
        self.feature_cols = []
        self.is_trained  = False

    # ── Load ──────────────────────────────
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV with 'date' and 'revenue' columns (+ optional extras)."""
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        print(f"✅ Loaded {len(df)} rows | Date range: {df['date'].min().date()} → {df['date'].max().date()}")
        return df

    # ── Preprocess ────────────────────────
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        df = create_time_features(df)
        df = create_lag_features(df)
        df = create_rolling_features(df)
        df = df.dropna().reset_index(drop=True)
        print(f"✅ After feature engineering: {len(df)} usable rows")
        return df

    # ── Train ─────────────────────────────
    def train(self, df: pd.DataFrame, target_col: str = "revenue"):
        """Train the model and return evaluation metrics."""
        self.feature_cols = [
            "month", "quarter", "month_sin", "month_cos", "is_q4",
            "revenue_lag_1", "revenue_lag_2", "revenue_lag_3",
            "rolling_mean_3", "rolling_mean_6", "rolling_std_3", "revenue_growth"
        ]
        # Add extra numeric columns if present (e.g. invoices, customers)
        extra = [c for c in df.columns if c not in self.feature_cols + [target_col, "date", "year"]]
        self.feature_cols += extra

        X = df[self.feature_cols]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Time-series: no shuffle
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc  = self.scaler.transform(X_test)

        self.model.fit(X_train_sc, y_train)
        self.is_trained = True

        # ── Metrics ──
        y_pred  = self.model.predict(X_test_sc)
        mae     = mean_absolute_error(y_test, y_pred)
        rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
        r2      = r2_score(y_test, y_pred)

        print("\n📊 Model Evaluation on Test Set:")
        print(f"   MAE  (Mean Absolute Error)  : ${mae:,.2f}")
        print(f"   RMSE (Root Mean Sq. Error)  : ${rmse:,.2f}")
        print(f"   R²   (Explained Variance)   : {r2:.4f}")

        self._X_test  = X_test
        self._y_test  = y_test
        self._y_pred  = y_pred
        self._df      = df
        self._target  = target_col

        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    # ── Predict Next Month ─────────────────
    def predict_next_month(self) -> float:
        """Predict next month's revenue using the most recent available data."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call .train() first.")

        df = self._df
        last_row = df.iloc[-1]

        next_features = {
            "month"          : (last_row["month"] % 12) + 1,
            "quarter"        : ((last_row["month"] % 12) // 3) + 1,
            "month_sin"      : np.sin(2 * np.pi * ((last_row["month"] % 12) + 1) / 12),
            "month_cos"      : np.cos(2 * np.pi * ((last_row["month"] % 12) + 1) / 12),
            "is_q4"          : int(((last_row["month"] % 12) + 1) in [10, 11, 12]),
            "revenue_lag_1"  : last_row[self._target],
            "revenue_lag_2"  : df.iloc[-2][self._target],
            "revenue_lag_3"  : df.iloc[-3][self._target],
            "rolling_mean_3" : df[self._target].iloc[-3:].mean(),
            "rolling_mean_6" : df[self._target].iloc[-6:].mean(),
            "rolling_std_3"  : df[self._target].iloc[-3:].std(),
            "revenue_growth" : (last_row[self._target] - df.iloc[-2][self._target]) / df.iloc[-2][self._target],
        }

        # Add extra cols if present
        extra = [c for c in self.feature_cols if c not in next_features]
        for c in extra:
            next_features[c] = last_row.get(c, 0)

        X_next = pd.DataFrame([next_features])[self.feature_cols]
        X_next_sc = self.scaler.transform(X_next)
        prediction = self.model.predict(X_next_sc)[0]

        print(f"\n🔮 Predicted Revenue for Next Month: ${prediction:,.2f}")
        return prediction

    # ── Feature Importance ────────────────
    def feature_importance(self):
        """Show which features drive the forecast most."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        coefs = pd.Series(self.model.coef_, index=self.feature_cols)
        coefs = coefs.abs().sort_values(ascending=False)
        print("\n📌 Top Feature Importances (by coefficient magnitude):")
        print(coefs.to_string())
        return coefs

    # ── Visualize ─────────────────────────
    def visualize(self, save_path: str = "outputs/forecast_plot.png"):
        """Plot actual vs predicted revenue + next month forecast."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Revenue Forecasting Dashboard", fontsize=16, fontweight="bold")

        # Plot 1: Full revenue trend
        ax1 = axes[0, 0]
        ax1.plot(self._df["date"], self._df[self._target], color="#1A56A4", linewidth=2, label="Actual Revenue")
        ax1.set_title("Historical Revenue Trend")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Revenue ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Actual vs Predicted (test set)
        ax2 = axes[0, 1]
        test_dates = self._df["date"].iloc[-len(self._y_test):]
        ax2.plot(test_dates, self._y_test.values, label="Actual",    color="#1A56A4", linewidth=2)
        ax2.plot(test_dates, self._y_pred,         label="Predicted", color="#E84545", linewidth=2, linestyle="--")
        ax2.set_title("Actual vs Predicted (Test Set)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Revenue ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Residuals
        ax3 = axes[1, 0]
        residuals = self._y_test.values - self._y_pred
        ax3.scatter(self._y_pred, residuals, color="#1A56A4", alpha=0.7)
        ax3.axhline(0, color="red", linestyle="--")
        ax3.set_title("Residual Plot")
        ax3.set_xlabel("Predicted Revenue")
        ax3.set_ylabel("Residuals")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Feature importance
        ax4 = axes[1, 1]
        coefs = pd.Series(self.model.coef_, index=self.feature_cols).abs().sort_values(ascending=True).tail(8)
        coefs.plot(kind="barh", ax=ax4, color="#1A56A4")
        ax4.set_title("Top 8 Feature Importances")
        ax4.set_xlabel("Coefficient Magnitude")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n✅ Plot saved to: {save_path}")
        plt.show()

    # ── Save / Load ───────────────────────
    def save_model(self, path: str = "outputs/model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_cols}, path)
        print(f"✅ Model saved to: {path}")

    def load_model(self, path: str = "outputs/model.pkl"):
        data = joblib.load(path)
        self.model        = data["model"]
        self.scaler       = data["scaler"]
        self.feature_cols = data["features"]
        self.is_trained   = True
        print(f"✅ Model loaded from: {path}")
