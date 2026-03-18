<<<<<<< HEAD
# FraudWatch — Real-Time Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![ML](https://img.shields.io/badge/ML-RandomForest-orange)
![Tests](https://img.shields.io/badge/Tests-13%20passing-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-284%2C807%20transactions-lightgrey)

A production-grade machine learning web application that detects fraudulent credit card transactions in real time — trained on 284,807 transactions mirroring the Kaggle Credit Card Fraud Detection dataset (0.172% fraud rate).

---

## Problem

Credit card fraud costs the global financial industry **$32 billion annually**. Traditional rule-based systems flag too many legitimate transactions (false positives) or miss novel fraud patterns. Machine learning enables dynamic, probabilistic detection at scale.

## Solution

FraudWatch uses a **Random Forest classifier** with class-weight balancing to handle extreme data imbalance, achieving industry-benchmark performance metrics. It provides:

- Instant fraud probability score per transaction  
- Risk level classification (SAFE / LOW / MEDIUM / HIGH)
- Top 3 contributing features explaining the decision
- Live transaction feed and analytics dashboard
- Actionable confidence scores for downstream decision systems

---

## Screenshots

### Transaction Checker
Submit a transaction and get instant fraud probability with risk breakdown.

### Analytics Dashboard  
Model performance metrics, feature importances, confusion matrix, class distribution.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn Random Forest |
| Backend | Python 3.11, Flask 3.0 |
| Frontend | HTML5, CSS3, Chart.js |
| Testing | Python unittest (13 tests) |
| Deployment | Render (PaaS), Gunicorn |
| Version Control | Git / GitHub |

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 Score | 1.0000 |
| Avg Precision | 1.0000 |

> **Note:** Perfect scores reflect synthetic training data. With the real Kaggle dataset, expect ROC-AUC ~0.97–0.99 (state of the art). See *Using Real Data* below.

---

## Project Structure

```
fraudwatch/
├── app.py                  # Flask app — routes, prediction logic, REST API
├── requirements.txt        # Dependencies
├── render.yaml             # One-click Render deployment
├── README.md
├── model/
│   ├── train.py            # Model training pipeline
│   ├── model.pkl           # Trained Random Forest (pre-built)
│   └── meta.pkl            # Model metadata (metrics, feature importances)
├── templates/
│   ├── index.html          # Transaction checker UI
│   └── dashboard.html      # Analytics dashboard
└── tests/
    └── test_app.py         # 13 unit tests (routes, predictions, edge cases)
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fraudwatch
cd fraudwatch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (only needed first time)
python model/train.py

# 4. Run the app
python app.py
```

Visit `http://localhost:5000`

---

## Run Tests

```bash
python -m unittest tests/test_app.py -v
```

13 tests covering: route loading, legitimate prediction, fraud prediction, missing features, invalid inputs, transaction logging, simulation endpoints, stats endpoint, risk level classification.

---

## Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — build and start commands are pre-configured
5. Deploy — get a live public URL in ~3 minutes

---

## Using the Real Kaggle Dataset

To train on the actual anonymised bank transaction data:

1. Download `creditcard.csv` from [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Place it in `model/`
3. In `model/train.py`, replace the data generation block with:

```python
df = pd.read_csv('model/creditcard.csv')
df['Amount_log'] = np.log1p(df['Amount'])
df['Hour'] = (df['Time'] / 3600).astype(int) % 24
```

4. Re-run `python model/train.py` — the model retrains on 284,807 real transactions

---

## API Reference

### `POST /api/predict`
Submit a transaction for fraud analysis.

**Request body (JSON):**
```json
{
  "V1": -4.8, "V2": 3.2, ..., "V28": 0.1,
  "Amount": 249.50,
  "Hour": 14
}
```

**Response:**
```json
{
  "transaction_id": "TXN482910",
  "fraud_probability": 94.3,
  "prediction": 1,
  "risk_level": "HIGH",
  "risk_color": "#e74c3c",
  "top_features": [
    {"feature": "V14", "importance": 18.35},
    {"feature": "V17", "importance": 12.10},
    {"feature": "V18", "importance": 9.87}
  ],
  "amount": 249.50,
  "timestamp": "14:32:11"
}
```

### `POST /api/simulate`
Generate a realistic random transaction.
```json
{ "fraud": true }   // or false for legitimate
```

### `GET /api/transactions`
Returns last 20 analysed transactions.

### `GET /api/stats`
Returns session-level statistics (total, fraud count, fraud rate).

---

## Social Impact

This system is designed to help:
- **Small banks and credit unions** that cannot afford enterprise fraud systems
- **Microfinance institutions** in developing markets where fraud disproportionately harms low-income borrowers
- **NGOs** that handle digital payments for social programs

---

## Author

**Preksha Agrawal** — B.Tech CS, KIIT University (CGPA: 9.2)  
[LinkedIn](https://linkedin.com/in/preksha-agrawal-ba0202321) · [GitHub](https://github.com/Preksha-Agrawal)
=======
# FraudWatch
>>>>>>> b0f5ae599c784c08a3cee513d3101c553339329f
