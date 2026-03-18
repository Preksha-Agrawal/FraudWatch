"""
FraudWatch - Unit Tests
Run with: python -m unittest tests/test_app.py -v
Also compatible with: pytest tests/test_app.py -v
"""
import unittest
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

def legit_payload():
    p = {f'V{i}': round(0.1 * (i % 3 - 1), 4) for i in range(1, 29)}
    p.update({'Amount': 45.0, 'Hour': 14})
    return p

def fraud_payload():
    p = {f'V{i}': 0.0 for i in range(1, 29)}
    p.update({
        'V1': -4.8, 'V2': 3.2, 'V3': -3.1, 'V4': 4.0,
        'V10': -5.0, 'V11': -5.5, 'V12': 4.8,
        'V14': -7.5, 'V17': -8.0, 'V18': -5.8,
        'Amount': 1.0, 'Hour': 2
    })
    return p

class TestFraudWatch(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def post_json(self, url, data):
        return self.client.post(url, data=json.dumps(data),
                                content_type='application/json')

    # ── ROUTE TESTS ──────────────────────────────────────────────────────────

    def test_index_loads(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'FraudWatch', r.data)

    def test_dashboard_loads(self):
        r = self.client.get('/dashboard')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'Dashboard', r.data)

    # ── PREDICTION TESTS ─────────────────────────────────────────────────────

    def test_predict_legitimate(self):
        r = self.post_json('/api/predict', legit_payload())
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn('fraud_probability', d)
        self.assertIn('prediction', d)
        self.assertIn('risk_level', d)
        self.assertEqual(d['prediction'], 0)
        self.assertLess(d['fraud_probability'], 50.0)

    def test_predict_fraud(self):
        r = self.post_json('/api/predict', fraud_payload())
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d['prediction'], 1)
        self.assertGreater(d['fraud_probability'], 50.0)

    def test_predict_missing_feature(self):
        r = self.post_json('/api/predict', {'Amount': 100.0, 'Hour': 10})
        self.assertEqual(r.status_code, 400)
        self.assertIn('error', r.get_json())

    def test_predict_invalid_value(self):
        payload = legit_payload()
        payload['V1'] = 'not_a_number'
        r = self.post_json('/api/predict', payload)
        self.assertEqual(r.status_code, 400)

    def test_predict_returns_top_features(self):
        r = self.post_json('/api/predict', legit_payload())
        d = r.get_json()
        self.assertIn('top_features', d)
        self.assertEqual(len(d['top_features']), 3)

    def test_predict_logs_transaction(self):
        self.post_json('/api/predict', legit_payload())
        r = self.client.get('/api/transactions')
        txns = r.get_json()
        self.assertGreaterEqual(len(txns), 1)
        self.assertIn('fraud_probability', txns[0])

    # ── SIMULATE TESTS ───────────────────────────────────────────────────────

    def test_simulate_legit(self):
        r = self.post_json('/api/simulate', {'fraud': False})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn('V1', d)
        self.assertIn('Amount', d)
        self.assertIn('Hour', d)

    def test_simulate_fraud(self):
        r = self.post_json('/api/simulate', {'fraud': True})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertIn('V14', d)
        self.assertLess(d['V14'], -4.0)

    # ── STATS TESTS ──────────────────────────────────────────────────────────

    def test_stats_endpoint(self):
        r = self.client.get('/api/stats')
        self.assertEqual(r.status_code, 200)

    def test_stats_after_predictions(self):
        self.post_json('/api/predict', legit_payload())
        self.post_json('/api/predict', fraud_payload())
        s = self.client.get('/api/stats').get_json()
        self.assertGreaterEqual(s['total'], 2)
        self.assertIn('fraud_rate', s)

    # ── RISK LEVEL TESTS ─────────────────────────────────────────────────────

    def test_risk_level_values(self):
        r = self.post_json('/api/predict', legit_payload())
        d = r.get_json()
        self.assertIn(d['risk_level'], ['SAFE', 'LOW', 'MEDIUM', 'HIGH'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
