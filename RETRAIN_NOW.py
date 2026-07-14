"""
RETRAIN_NOW.py — Run from C:\heart_disease_app\
Command: python RETRAIN_NOW.py
"""
import numpy as np
import pickle, os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
N = 600

def make_patients(n, risk_level='low'):
    r = np.random.RandomState({'high':42,'medium':77,'low':99}[risk_level])
    if risk_level == 'high':
        age      = r.randint(55, 77, n)
        sex      = r.choice([0,1], n, p=[0.15, 0.85])
        cp       = r.choice([0,1,2,3], n, p=[0.02, 0.05, 0.10, 0.83])
        trestbps = r.randint(145, 200, n)
        chol     = r.randint(245, 400, n)
        fbs      = r.choice([0,1], n, p=[0.25, 0.75])
        restecg  = r.choice([0,1,2], n, p=[0.10, 0.60, 0.30])
        thalach  = r.randint(88, 130, n)
        exang    = r.choice([0,1], n, p=[0.10, 0.90])
        oldpeak  = np.round(r.uniform(2.0, 6.2, n), 1)
        slope    = r.choice([0,1,2], n, p=[0.05, 0.65, 0.30])
        ca       = r.choice([0,1,2,3], n, p=[0.05, 0.20, 0.35, 0.40])
        thal     = r.choice([1,2,3], n, p=[0.03, 0.15, 0.82])
        target   = np.full(n, 2, dtype=int)        # class 2 = high risk
    elif risk_level == 'medium':
        age      = r.randint(45, 62, n)
        sex      = r.choice([0,1], n, p=[0.40, 0.60])
        cp       = r.choice([0,1,2,3], n, p=[0.10, 0.30, 0.40, 0.20])
        trestbps = r.randint(128, 150, n)
        chol     = r.randint(210, 260, n)
        fbs      = r.choice([0,1], n, p=[0.60, 0.40])
        restecg  = r.choice([0,1,2], n, p=[0.35, 0.50, 0.15])
        thalach  = r.randint(120, 155, n)
        exang    = r.choice([0,1], n, p=[0.50, 0.50])
        oldpeak  = np.round(r.uniform(0.8, 2.2, n), 1)
        slope    = r.choice([0,1,2], n, p=[0.25, 0.55, 0.20])
        ca       = r.choice([0,1,2,3], n, p=[0.30, 0.45, 0.18, 0.07])
        thal     = r.choice([1,2,3], n, p=[0.20, 0.50, 0.30])
        target   = np.full(n, 1, dtype=int)        # class 1 = medium/borderline
    else:
        age      = r.randint(25, 50, n)
        sex      = r.choice([0,1], n, p=[0.60, 0.40])
        cp       = r.choice([0,1,2,3], n, p=[0.60, 0.25, 0.12, 0.03])
        trestbps = r.randint(88, 128, n)
        chol     = r.randint(130, 210, n)
        fbs      = r.choice([0,1], n, p=[0.90, 0.10])
        restecg  = r.choice([0,1,2], n, p=[0.75, 0.18, 0.07])
        thalach  = r.randint(155, 202, n)
        exang    = r.choice([0,1], n, p=[0.90, 0.10])
        oldpeak  = np.round(r.uniform(0.0, 0.9, n), 1)
        slope    = r.choice([0,1,2], n, p=[0.65, 0.28, 0.07])
        ca       = r.choice([0,1,2,3], n, p=[0.78, 0.15, 0.05, 0.02])
        thal     = r.choice([1,2,3], n, p=[0.70, 0.22, 0.08])
        target   = np.full(n, 0, dtype=int)        # class 0 = low/no risk

    X = np.column_stack([age,sex,cp,trestbps,chol,fbs,restecg,
                         thalach,exang,oldpeak,slope,ca,thal])
    return X, target

Xh, yh = make_patients(N, 'high')
Xm, ym = make_patients(N, 'medium')
Xl, yl = make_patients(N, 'low')

X = np.vstack([Xh, Xm, Xl])
y = np.concatenate([yh, ym, yl])

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Classes → Low(0):{sum(y==0)}  Medium(1):{sum(y==1)}  High(2):{sum(y==2)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
Xtr = scaler.fit_transform(X_train)
Xte = scaler.transform(X_test)

lr = LogisticRegression(max_iter=3000, random_state=42, C=1.0)
rf  = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
svm = SVC(kernel='rbf', probability=True, C=10, gamma='scale', random_state=42)

lr.fit(Xtr, y_train);  print(f"LR  accuracy: {accuracy_score(y_test, lr.predict(Xte))*100:.1f}%")
rf.fit(Xtr, y_train);  print(f"RF  accuracy: {accuracy_score(y_test, rf.predict(Xte))*100:.1f}%")
svm.fit(Xtr, y_train); print(f"SVM accuracy: {accuracy_score(y_test, svm.predict(Xte))*100:.1f}%")

# ── Sanity check ──────────────────────────────────────────────────────────────
HIGH   = np.array([[67,1,3,160,286,1,1,108,1,2.5,1,3,3]])
MEDIUM = np.array([[52,1,1,138,230,0,1,132,1,1.2,1,1,2]])
LOW    = np.array([[32,0,0,115,175,0,0,178,0,0.0,0,0,1]])

for name, sample, expected in [('HIGH',HIGH,2),('MEDIUM',MEDIUM,1),('LOW',LOW,0)]:
    s = scaler.transform(sample)
    preds = [lr.predict(s)[0], rf.predict(s)[0], svm.predict(s)[0]]
    ok = '✅' if all(p==expected for p in preds) else '⚠️'
    print(f"{ok} {name:6} → LR:{preds[0]} RF:{preds[1]} SVM:{preds[2]}  (expected {expected})")

os.makedirs('models', exist_ok=True)
with open('models/scaler.pkl','wb') as f:               pickle.dump(scaler,f)
with open('models/logistic_regression_model.pkl','wb') as f: pickle.dump(lr,f)
with open('models/random_forest_model.pkl','wb') as f:  pickle.dump(rf,f)
with open('models/svm_model.pkl','wb') as f:            pickle.dump(svm,f)
with open('models/feature_names.pkl','wb') as f:        pickle.dump(['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'],f)

print("\n✅ All models saved to models/  — restart app.py now!")
