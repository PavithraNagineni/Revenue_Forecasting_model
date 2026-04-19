# 📈 Revenue Forecasting Model — Linear Regression + Time Series

**Author:** Pavithra Nagineni  
**Tech Stack:** Python · Pandas · NumPy · Scikit-learn · FastAPI · Matplotlib  
**Purpose:** Predicts next month's revenue using historical billing data, feature engineering, and Linear Regression — aligned with real-world FP&A workflows.

---

## 🧠 What This Project Does

- Ingests monthly revenue + billing data (real or synthetic)
- Engineers time-series features: lag values, rolling averages, seasonality encoding
- Trains **Linear Regression** (+ Ridge option) to predict next month's revenue
- Evaluates with MAE, RMSE, and R² metrics
- Generates a **4-panel forecast dashboard** (trend, actual vs predicted, residuals, feature importance)
- Exposes predictions via a **FastAPI REST endpoint**

---

## 📁 Project Structure

```
revenue_forecasting/
├── main.py                   # ← Run this first
├── api.py                    # FastAPI service
├── requirements.txt
├── data/
│   └── revenue_data.csv      # Auto-generated on first run
├── models/
│   └── revenue_model.py      # Core ML pipeline
├── utils/
│   └── data_generator.py     # Synthetic data generator
└── outputs/
    ├── forecast_plot.png     # Auto-generated dashboard
    └── model.pkl             # Saved model
```

---

## ⚡ Quick Start

### Step 1 — Clone / Download
```bash
git clone https://github.com/PavithraNagineni/revenue-forecasting
cd revenue_forecasting
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the Model
```bash
# Option A: Use synthetic data (auto-generated)
python main.py

# Option B: Use your own CSV (must have 'date' and 'revenue' columns)
python main.py --data path/to/your_data.csv

# Option C: Use Ridge Regression instead
python main.py --model ridge
```

### Step 4 — View Results
- **Terminal:** MAE, RMSE, R², next month prediction
- **Plot:** `outputs/forecast_plot.png`
- **Model:** `outputs/model.pkl`

---

## 🌐 Run the API (Optional)

```bash
# First train and save the model
python main.py

# Then start the API server
uvicorn api:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Example API Call
```bash
curl -X POST "http://localhost:8000/forecast/custom" \
  -H "Content-Type: application/json" \
  -d '{
    "month": 5,
    "quarter": 2,
    "revenue_lag_1": 145000,
    "revenue_lag_2": 138000,
    "revenue_lag_3": 132000,
    "rolling_mean_3": 138333,
    "rolling_mean_6": 135000,
    "rolling_std_3": 5200,
    "revenue_growth": 0.05
  }'
```

**Response:**
```json
{
  "predicted_revenue": 151234.56,
  "model_type": "LinearRegression",
  "currency": "USD",
  "note": "Prediction based on provided features using trained Linear Regression model."
}
```

---

## 📊 Input Data Format

Your CSV should have at minimum:

| date       | revenue   |
|------------|-----------|
| 2022-01-01 | 102500.00 |
| 2022-02-01 | 98300.00  |
| 2022-03-01 | 110200.00 |

Optional extra columns (improve accuracy):
- `invoices` — number of invoices processed that month
- `customers` — number of active customers
- `avg_deal_size` — average invoice value

---

## 🔬 Feature Engineering

| Feature | Description |
|---------|-------------|
| `revenue_lag_1/2/3` | Revenue from 1, 2, 3 months ago |
| `rolling_mean_3/6` | 3-month and 6-month rolling average |
| `rolling_std_3` | 3-month rolling standard deviation |
| `month_sin / month_cos` | Cyclical month encoding |
| `is_q4` | Q4 seasonality flag (Oct–Dec) |
| `revenue_growth` | Month-over-month % change |

---

## 📈 Sample Output

```
✅ Loaded 36 rows | Date range: 2022-01-01 → 2024-12-01
✅ After feature engineering: 33 usable rows

📊 Model Evaluation on Test Set:
   MAE  (Mean Absolute Error)  : $4,231.18
   RMSE (Root Mean Sq. Error)  : $5,102.44
   R²   (Explained Variance)   : 0.9612

🔮 Predicted Revenue for Next Month: $187,432.50
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **Pandas** | Data loading, transformation, feature engineering |
| **NumPy** | Numerical operations, cyclical encoding |
| **Scikit-learn** | Linear Regression, Ridge, StandardScaler, metrics |
| **Matplotlib/Seaborn** | Forecast dashboard visualization |
| **FastAPI** | REST API for serving predictions |
| **Joblib** | Model serialization (save/load) |

---

## 💡 Relevance to FP&A

This project directly maps to real FP&A workflows:
- **Automated Data Pipelines** → Pandas preprocessing pipeline
- **Statistical Forecasting** → Linear Regression + Time-Series features
- **Predictive Cost/Revenue** → Next month prediction from 250+ invoices pattern
- **Variance Attribution** → Residual analysis + feature importance
- **API Integration** → FastAPI endpoint for ERP/dashboard integration
"# Revenue-forecasting-model-" 
