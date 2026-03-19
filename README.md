# FraudWatch — Real-Time Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![ML](https://img.shields.io/badge/ML-Random%20Forest-orange)

A production-grade machine learning web application that detects fraudulent 
credit card transactions in real time — trained on 284,807 transactions 
mirroring the Kaggle Credit Card Fraud Detection dataset (0.172% fraud rate).

🔗 **Live Demo:** https://your-render-url.onrender.com

## Problem
Credit card fraud costs the global financial industry $32 billion annually. 
Traditional rule-based systems flag too many legitimate transactions (false 
positives) or miss novel fraud patterns. Machine learning enables dynamic, 
probabilistic detection at scale.

## Solution
FraudWatch uses a Random Forest classifier with class-weight balancing to 
handle extreme data imbalance. It provides:

- Instant fraud probability score per transaction
- Risk level classification (SAFE / LOW / MEDIUM / HIGH)
- Top 3 contributing features explaining the decision
- Live transaction feed and analytics dashboard
- Actionable confidence scores for downstream decision systems

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn Random Forest |
| Backend | Python 3.11, Flask 3.0 |
| Frontend | HTML5, CSS3, Chart.js |
| Testing | Python unittest (13 tests) |
| Deployment | Render (PaaS), Gunicorn |
| Version Control | Git / GitHub |

## Model Performance (Real Kaggle Dataset)

| Metric | Score |
|---|---|
| ROC-AUC | ~0.97-0.99 |
| Precision | ~0.95 |
| Recall | ~0.92 |
| F1 Score | ~0.93 |

> Trained on 284,807 real transactions with 0.172% fraud rate.
> Class imbalance handled via Random Forest class-weight balancing.

## Project Structure
```
fraudwatch/
├── app.py                  # Flask app — routes, prediction, REST API
├── requirements.txt        # Dependencies
├── render.yaml             # One-click Render deployment
├── README.md
├── model/
│   ├── train.py            # Model training pipeline
│   ├── model.pkl           # Trained Random Forest
│   └── meta.pkl            # Model metadata
├── templates/
│   ├── index.html          # Transaction checker UI
│   └── dashboard.html      # Analytics dashboard
└── tests/
    └── test_app.py         # 13 unit tests
```

## Run Locally
```bash
git clone https://github.com/Preksha-Agrawal/fraudwatch
cd fraudwatch
pip install -r requirements.txt
python model/train.py
python app.py
```
Visit http://localhost:5000

## Run Tests
```bash
python -m unittest tests/test_app.py -v
```
13 tests covering: route loading, legitimate prediction, fraud prediction, 
missing features, invalid inputs, transaction logging, simulation endpoints, 
stats endpoint, risk level classification.

## Deploy on Render
1. Push repo to GitHub
2. Go to render.com → New Web Service
3. Connect your GitHub repo
4. Render auto-detects render.yaml
5. Deploy — live public URL in ~3 minutes

## API Reference

### POST /api/predict
```json
{
  "V1": -4.8, "V2": 3.2, "V28": 0.1,
  "Amount": 249.50,
  "Hour": 14
}
```
Response:
```json
{
  "transaction_id": "TXN482910",
  "fraud_probability": 94.3,
  "prediction": 1,
  "risk_level": "HIGH",
  "top_features": [
    {"feature": "V14", "importance": 18.35}
  ],
  "timestamp": "14:32:11"
}
```

## Social Impact
Designed to help small banks, credit unions and microfinance institutions 
that cannot afford enterprise fraud systems — particularly in developing 
markets where fraud disproportionately harms low-income borrowers.

## Author
**Preksha Agrawal** — B.Tech CS, KIIT University (CGPA: 9.2)  
[LinkedIn](http://www.linkedin.com/in/preksha-agrawal-ba0202321) · 
[GitHub](https://github.com/Preksha-Agrawal)
