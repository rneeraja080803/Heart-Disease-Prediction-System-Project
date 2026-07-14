# ❤️ CardioSense — Heart Disease Prediction System

A Machine Learning web application that predicts cardiovascular 
risk as 🟢 Low, 🟡 Medium (Borderline), or 🔴 High using 
13 clinical parameters.

## Models Used
- Logistic Regression — 85% accuracy
- Random Forest — 97.29% accuracy  
- SVM (RBF kernel) — 89.05% accuracy

## Tech Stack
Python · Flask · Scikit-learn · HTML · CSS · Pickle

## How to Run
1. pip install -r requirements.txt
2. python RETRAIN_NOW.py
3. python app.py
4. Open index.html with Live Server

## Features
- 3-level colour-coded risk prediction (Green/Yellow/Red)
- Probability bars for all 3 risk classes
- Consensus voting from all 3 models
- 13 clinical input fields with tooltips
- Medical disclaimer included

## Project Structure
heart_disease_app/
├── app.py
├── index.html
├── RETRAIN_NOW.py
├── requirements.txt
└── models/
    ├── scaler.pkl
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    └── svm_model.pkl
