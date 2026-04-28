import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # 1. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 2. Handle missing values
    df.dropna(subset=["Test Results"], inplace=True)
    df["Billing Amount"] = df["Billing Amount"].fillna(df["Billing Amount"].median())

    # 3. Convert date columns
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"]    = pd.to_datetime(df["Discharge Date"])
    df["Length of Stay"]    = (df["Discharge Date"] - df["Date of Admission"]).dt.days

    # 4. Standardize categorical values
    cat_cols = ["Gender","Blood Type","Medical Condition",
                "Admission Type","Insurance Provider","Medication","Test Results"]
    for col in cat_cols:
        df[col] = df[col].str.strip().str.title()

    # 5. Drop irrelevant columns
    df.drop(columns=["Name","Doctor","Hospital","Room Number"], inplace=True)

    # 6. Encode target label
    df["Target"] = df["Test Results"].map(
        {"Normal": 0, "Abnormal": 1, "Inconclusive": 2}
    )

    return df

if __name__ == "__main__":
    raw   = Path("data/raw/healthcare_dataset.csv")
    clean = Path("data/processed/cleaned.csv")
    df    = clean_data(str(raw))
    df.to_csv(clean, index=False)
    print(f"Cleaned data saved: {len(df)} rows")