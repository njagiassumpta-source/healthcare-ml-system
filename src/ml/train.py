import os, joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier
from datetime import datetime

DATABASE_URL = "postgresql://health_user:health123@localhost:5432/healthcare_db"
engine = create_engine(DATABASE_URL)

FEATURES = ["age","gender","blood_type","medical_condition",
            "billing_amount","admission_type","insurance_provider",
            "medication","length_of_stay"]
TARGET = "target"

def preprocess(df):
    cat_cols = ["gender","blood_type","medical_condition",
                "admission_type","insurance_provider","medication"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds, average='weighted'):.4f}")
    print(f"Recall   : {recall_score(y_test, preds, average='weighted'):.4f}")
    print(f"F1 Score : {f1_score(y_test, preds, average='weighted'):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, preds)}")
    return f1_score(y_test, preds, average='weighted')

def train():
    print("Reading data from database...")
    df = pd.read_sql("SELECT * FROM patients", engine)
    df = preprocess(df[FEATURES + [TARGET]].dropna())
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "XGBoost": XGBClassifier(n_estimators=200,
                                 eval_metric="mlogloss", random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    }

    best_score, best_model, best_name = 0, None, ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = evaluate(name, model, X_test, y_test)
        if score > best_score:
            best_score, best_model, best_name = score, model, name

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"models/best_model_{ts}.pkl"
    joblib.dump(best_model, path)
    joblib.dump(best_model, "models/latest_model.pkl")
    print(f"\nBest model: {best_name} (F1={best_score:.4f}), saved to {path}")

if __name__ == "__main__":
    train()