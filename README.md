# Healthcare ML System

A machine learning system that predicts patient test results.

## Setup
1. Install UV: `pip install uv`
2. Install dependencies: `uv sync`
3. Run API: `uvicorn src.api.main:app --reload`

## API Usage
POST /predict
```json
{
  "Age": 55,
  "Gender": "Male",
  "Blood_Type": "A+",
  "Medical_Condition": "Diabetes",
  "Billing_Amount": 12345,
  "Admission_Type": "Emergency",
  "Insurance_Provider": "Cigna",
  "Medication": "Aspirin"
}
```

## Response
```json
{"predicted_test_result": "Abnormal"}
```

## Live API
https://healthcare-ml-system-XXXX.onrender.com