"""
FraudWatch - Real-Time Credit Card Fraud Detection
Flask application with prediction API and analytics dashboard.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import json
from datetime import datetime
import random

app = Flask(__name__)

BASE = os.path.dirname(__file__)

# Load model and metadata
with open(os.path.join(BASE, 'model', 'model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE, 'model', 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)

FEATURES = meta['features']
V_FEATURES = [f for f in FEATURES if f.startswith('V')]

# In-memory transaction log (resets on restart; use DB in production)
transaction_log = []

def make_transaction_id():
    return 'TXN' + str(random.randint(100000, 999999))

def classify_risk(prob):
    if prob >= 0.80:
        return 'HIGH', '#e74c3c'
    elif prob >= 0.40:
        return 'MEDIUM', '#f39c12'
    elif prob >= 0.10:
        return 'LOW', '#f1c40f'
    else:
        return 'SAFE', '#27ae60'

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', meta=meta)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Build feature vector
        row = {}
        for v in V_FEATURES:
            val = data.get(v)
            if val is None:
                return jsonify({'error': f'Missing feature: {v}'}), 400
            try:
                row[v] = float(val)
            except (ValueError, TypeError):
                return jsonify({'error': f'Invalid value for {v}'}), 400

        amount = float(data.get('Amount', 0))
        row['Amount_log'] = np.log1p(amount)
        row['Hour'] = int(data.get('Hour', 12)) % 24

        import pandas as pd
        X = pd.DataFrame([row])[FEATURES]

        prob = float(model.predict_proba(X)[0][1])
        pred = int(prob >= 0.5)
        risk_level, risk_color = classify_risk(prob)

        # Get top contributing features from forest
        rf = model.named_steps['clf']
        importances = rf.feature_importances_
        contrib = {FEATURES[i]: abs(float(row.get(FEATURES[i], 0)) * importances[i])
                   for i in range(len(FEATURES))}
        top3 = sorted(contrib.items(), key=lambda x: x[1], reverse=True)[:3]
        top_features = [{'feature': k, 'importance': round(v * 100, 2)} for k, v in top3]

        # Log transaction
        txn = {
            'id': make_transaction_id(),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'amount': round(amount, 2),
            'fraud_probability': round(prob * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'prediction': pred,
            'top_features': top_features
        }
        transaction_log.insert(0, txn)
        if len(transaction_log) > 50:
            transaction_log.pop()

        return jsonify({
            'transaction_id': txn['id'],
            'fraud_probability': round(prob * 100, 2),
            'prediction': pred,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'top_features': top_features,
            'amount': round(amount, 2),
            'timestamp': txn['timestamp']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """Generate a realistic random transaction (fraud or legit)."""
    try:
        data = request.get_json() or {}
        force_fraud = data.get('fraud', False)

        if force_fraud:
            # Fraud-like values matching training distribution
            v_vals = {f'V{i}': round(np.random.randn(), 4) for i in range(1, 29)}
            v_vals['V1']  = round(np.random.normal(-4.8, 1.2), 4)
            v_vals['V2']  = round(np.random.normal(3.2, 1.0), 4)
            v_vals['V3']  = round(np.random.normal(-3.1, 1.0), 4)
            v_vals['V4']  = round(np.random.normal(4.0, 1.0), 4)
            v_vals['V10'] = round(np.random.normal(-5.0, 1.0), 4)
            v_vals['V11'] = round(np.random.normal(-5.5, 1.0), 4)
            v_vals['V12'] = round(np.random.normal(4.8, 1.0), 4)
            v_vals['V14'] = round(np.random.normal(-7.5, 1.5), 4)
            v_vals['V17'] = round(np.random.normal(-8.0, 1.5), 4)
            v_vals['V18'] = round(np.random.normal(-5.8, 1.0), 4)
            amount = round(abs(np.random.lognormal(3.5, 1.2)), 2)
        else:
            v_vals = {f'V{i}': round(np.random.randn() * [2.0,1,1.5,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1][i-1], 4)
                      for i in range(1, 29)}
            amount = round(abs(np.random.lognormal(4.0, 1.5)), 2)

        hour = random.randint(0, 23)
        v_vals['Amount'] = amount
        v_vals['Hour'] = hour

        return jsonify(v_vals)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    return jsonify(transaction_log[:20])


@app.route('/api/stats', methods=['GET'])
def get_stats():
    if not transaction_log:
        return jsonify({'total': 0, 'fraud': 0, 'safe': 0, 'avg_prob': 0})
    total = len(transaction_log)
    fraud = sum(1 for t in transaction_log if t['prediction'] == 1)
    avg_prob = round(sum(t['fraud_probability'] for t in transaction_log) / total, 2)
    return jsonify({
        'total': total,
        'fraud': fraud,
        'safe': total - fraud,
        'avg_prob': avg_prob,
        'fraud_rate': round(fraud / total * 100, 1)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
