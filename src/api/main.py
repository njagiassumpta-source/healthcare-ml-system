import joblib, numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Healthcare Prediction API", version="1.0")

MODEL_PATH = Path("models/latest_model.pkl")

class PatientInput(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: str
    Insurance_Provider: str
    Medication: str

LABEL_MAPS = {
    "Gender": ["Female","Male","Other"],
    "Blood_Type": ["A+","A-","AB+","AB-","B+","B-","O+","O-"],
    "Medical_Condition": ["Arthritis","Asthma","Cancer","Diabetes",
                          "Hypertension","Obesity"],
    "Admission_Type": ["Elective","Emergency","Urgent"],
    "Insurance_Provider": ["Aetna","Blue Cross","Cigna","Medicare","UnitedHealthCare"],
    "Medication": ["Aspirin","Ibuprofen","Lipitor","Paracetamol","Penicillin"],
}
RESULT_MAP = {0: "Normal", 1: "Abnormal", 2: "Inconclusive"}

def encode(val, col):
    try:
        return LABEL_MAPS[col].index(val)
    except ValueError:
        return 0

@app.get("/")
def root():
    return {"status": "Healthcare Prediction API is running"}

@app.post("/predict")
def predict(patient: PatientInput):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not yet trained. Run training first.")
    model = joblib.load(MODEL_PATH)
    features = np.array([[
        patient.Age,
        encode(patient.Gender, "Gender"),
        encode(patient.Blood_Type, "Blood_Type"),
        encode(patient.Medical_Condition, "Medical_Condition"),
        patient.Billing_Amount,
        encode(patient.Admission_Type, "Admission_Type"),
        encode(patient.Insurance_Provider, "Insurance_Provider"),
        encode(patient.Medication, "Medication"),
        0
    ]])
    pred = model.predict(features)[0]
    return {"predicted_test_result": RESULT_MAP[int(pred)]}