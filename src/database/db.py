import os
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://health_user:health123@localhost:5432/healthcare_db"
engine = create_engine(DATABASE_URL)


engine = create_engine(DATABASE_URL)

def create_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY,
                age INTEGER,
                gender VARCHAR(10),
                blood_type VARCHAR(5),
                medical_condition VARCHAR(100),
                date_of_admission DATE,
                discharge_date DATE,
                length_of_stay INTEGER,
                admission_type VARCHAR(20),
                insurance_provider VARCHAR(100),
                billing_amount FLOAT,
                medication VARCHAR(100),
                test_results VARCHAR(20),
                target INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    df.to_sql("patients_staging", engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO patients (age, gender, blood_type, medical_condition,
                date_of_admission, discharge_date, length_of_stay, admission_type,
                insurance_provider, billing_amount, medication, test_results, target)
            SELECT age, gender, blood_type, medical_condition,
                date_of_admission::date, discharge_date::date, length_of_stay,
                admission_type, insurance_provider, billing_amount, medication,
                test_results, target
            FROM patients_staging
            ON CONFLICT DO NOTHING
        """))
        conn.commit()
    print("Data loaded into PostgreSQL.")

if __name__ == "__main__":
    create_table()
    load_data("data/processed/cleaned.csv")