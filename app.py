from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle, numpy as np, os

app = Flask(__name__, static_folder='.')
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(BASE, 'models', name), 'rb') as f:
        return pickle.load(f)

try:
    scaler    = load('scaler.pkl')
    lr_model  = load('logistic_regression_model.pkl')
    rf_model  = load('random_forest_model.pkl')
    svm_model = load('svm_model.pkl')
    print(f"✅ All models loaded. Scaler expects {scaler.n_features_in_} features.")
except Exception as e:
    print(f"❌ Model load error: {e}")
    scaler = lr_model = rf_model = svm_model = None

FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg',
            'thalach','exang','oldpeak','slope','ca','thal']

# 0 = Low/No Risk   1 = Medium/Borderline   2 = High Risk
RISK_LABEL = {0: 'No Heart Disease', 1: 'Borderline Risk', 2: 'Heart Disease Detected'}
RISK_LEVEL = {0: 'low', 1: 'medium', 2: 'high'}

def predict_all(features):
    arr = np.array(features, dtype=float).reshape(1, -1)
    arr_sc = scaler.transform(arr)

    def run(model, name):
        pred     = int(model.predict(arr_sc)[0])
        proba    = model.predict_proba(arr_sc)[0]
        all_prob = {RISK_LABEL[i]: round(float(p)*100,1) for i,p in enumerate(proba)}
        return {
            'model':            name,
            'prediction':       pred,
            'risk_level':       RISK_LEVEL[pred],
            'label':            RISK_LABEL[pred],
            'confidence':       round(float(proba[pred])*100, 1),
            'all_probabilities': all_prob
        }

    return [
        run(lr_model,  'Logistic Regression'),
        run(rf_model,  'Random Forest'),
        run(svm_model, 'Support Vector Machine'),
    ]

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if None in (scaler, lr_model, rf_model, svm_model):
        return jsonify({'error': 'Models not loaded. Run RETRAIN_NOW.py first.'}), 500
    data = request.get_json()
    try:
        features = [float(data[f]) for f in FEATURES]
    except (KeyError, TypeError) as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    try:
        return jsonify({'results': predict_all(features)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)