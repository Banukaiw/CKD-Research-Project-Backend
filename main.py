import uvicorn
from fastapi import FastAPI,Depends,File, UploadFile
import pandas as pd
import pickle
from model_loader import load_model,DecisionTree_model,Nephrotic_model,CKD_model,AKI_model
from prophet import Prophet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

def get_loaded_model():
    model = load_model()
    return model

def get_DecisionTree_model():
    model = DecisionTree_model()
    return model

@app.on_event("startup")
async def startup_event():
    app.loaded_model = get_loaded_model()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict_ckd")
async def predict_ckd(
    age: int,
    blood_pressure: float,
    specific_gravity: float,
    albumin: float,
    sugar: float,
    red_blood_cells: int,
    pus_cell: int,
    pus_cell_clumps: int,
    bacteria: int,
    blood_glucose_random: float,
    blood_urea: float,
    serum_creatinine: float,
    sodium: float,
    potassium: float,
    haemoglobin: float,
    packed_cell_volume: float,
    white_blood_cell_count: float,
    red_blood_cell_count: float,
    hypertension: int,
    diabetes_mellitus: int,
    coronary_artery_disease: int,
    appetite: int,
    peda_edema: int,
    aanemia: int,
    loaded_model=Depends(get_loaded_model)
):
    data = pd.DataFrame({
        'age': [age],
        'blood_pressure': [blood_pressure],
        'specific_gravity': [specific_gravity],
        'albumin': [albumin],
        'sugar': [sugar],
        'red_blood_cells': [red_blood_cells],
        'pus_cell': [pus_cell],
        'pus_cell_clumps': [pus_cell_clumps],
        'bacteria': [bacteria],
        'blood_glucose_random': [blood_glucose_random],
        'blood_urea': [blood_urea],
        'serum_creatinine': [serum_creatinine],
        'sodium': [sodium],
        'potassium': [potassium],
        'haemoglobin': [haemoglobin],
        'packed_cell_volume': [packed_cell_volume],
        'white_blood_cell_count': [white_blood_cell_count],
        'red_blood_cell_count': [red_blood_cell_count],
        'hypertension': [hypertension],
        'diabetes_mellitus': [diabetes_mellitus],
        'coronary_artery_disease': [coronary_artery_disease],
        'appetite': [appetite],
        'peda_edema': [peda_edema],
        'aanemia': [aanemia]
    })

    new_pred_rf_clf = loaded_model.predict(data)
    return int(new_pred_rf_clf)


def Drg_forecast(df):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(3, freq='MS')
    forecast = m.predict(future)
    forecast = forecast[['ds','yhat']].tail(3)
    # forecast.to_csv("last3.csv")
    result = []
    for index, row in forecast.iterrows():
        timestamp = pd.Timestamp(row['ds'])
        date_only = timestamp.date()
        rounded_yhat = round(row['yhat'])
        result.append({'date': str(date_only), 'prediction': rounded_yhat})
    return result


@app.post("/uploadcsv/")
def upload_csv(csv_file: UploadFile = File(...)):
    dataframe = pd.read_csv(csv_file.file)
    dataframe.columns = ['ds', 'y']
    predictions = Drg_forecast(dataframe)
    print(predictions)
    return predictions

@app.post("/forecast")
def forecast_via_values(date:str,sales:int):
    data = pd.DataFrame({'ds':[date],'y':[sales]})
    df = pd.read_csv('drg1.csv')
    df.columns = ['ds', 'y']
    frames = [df, data]
    dataframe = pd.concat(frames)
    predictions = Drg_forecast(dataframe)
    print(predictions)
    return predictions

def recommend_diet(patient_data,loaded_model):  
    patient_df = pd.DataFrame([patient_data])
    
    gender_mapping = {'M': 0, 'F': 1}
    conditions_mapping = {'Hypertension': 0, 'Diabetes': 1, 'nan': 2}
    food_mapping = {'Low Sodium': 0, 'Low Potassium': 1, 'Balanced': 2}

    patient_df['Gender'] = patient_df['Gender'].map(gender_mapping)
    patient_df['Other Conditions'] = patient_df['Other Conditions'].map(conditions_mapping)
    patient_df['Preferred Food'] = patient_df['Preferred Food'].map(food_mapping)

    # Predict the kidney disease stage
    level = loaded_model.predict(patient_df)[0]
    print(level)

    if level == 1:
        return "Maintain a balanced diet. No specific restrictions but ensure a healthy intake of protein, sodium, potassium and phosphorus."
    elif level == 2:
        return "Start to monitor and limit the intake of protein, sodium, potassium and phosphorus to prevent further kidney damage."
    elif level == 3:
        return "Follow a diet low in protein, sodium, potassium and phosphorus. Consult a dietitian for a personalized diet plan."

@app.post("/Suggest_Diet_Plan")
def Suggest_Diet_Plan(
    Age: int,
    Gender: str,
    BMI: int,
    Current_Protein_Intake: int,
    Current_Sodium_Intake: int,
    Current_Potassium_Intake: int,
    Current_Phosphorus_Intake: int,
    Other_Conditions: str,
    GFR: int,
    Proteinuria: int,
    Preferred_Food: str,
    loaded_model=Depends(get_DecisionTree_model)
):
    patient_data = {
    'Age': Age,
    'Gender': Gender,
    'BMI': BMI,
    'Current Protein Intake (g)': Current_Protein_Intake,
    'Current Sodium Intake (mg)': Current_Sodium_Intake,
    'Current Potassium Intake (mg)': Current_Potassium_Intake,
    'Current Phosphorus Intake (mg)': Current_Phosphorus_Intake,
    'Other Conditions': Other_Conditions,
    'GFR': GFR,
    'Proteinuria':Proteinuria,
    'Preferred Food': Preferred_Food,
    }
    suggesting = recommend_diet(patient_data,loaded_model)
    return suggesting

class Symptoms(BaseModel):
    Age: Optional[int] 
    Gender: str
    AKIDiagnosis: str
    AKISeverity: str
    InitialCreatinine: float
    PeakCreatinine: float
    UrineOutput: int
    # Swelling: str,
    HasDisease: str

    Underlying_Cause: str
    Proteinuria: str
    Edema: str
    Albumin_Level: str
    Cholesterol_Level: str
    Infection_Count: int
    Nephrotic_Syndrome: str

    Change_in_Urination: bool
    Swelling: bool
    Fatigue_Weakness: bool
    Skin_Rash_Itching: bool
    Shortness_of_Breath: bool
    Metallic_Taste_in_Mouth: bool
    Feeling_Cold: bool
    Dizziness_Trouble_Concentrating: bool
    Pain_in_Back_or_Sides: bool
    Nausea_Vomiting: bool
    Disease: str

@app.post("/predict_diseas")
def predict_disease(
    Age: int,
    Gender: Optional[str] = None,

    AKIDiagnosis: Optional[str] = None,
    InitialCreatinine: Optional[float] = None,
    PeakCreatinine: Optional[float] = None,
    UrineOutput: Optional[int] = None,
    Swelling: Optional[bool] = None,

    Proteinuria: Optional[str] = None,
    Edema: Optional[str] = None,
    Albumin_Level: Optional[str] = None,

    Change_in_Urination: Optional[bool] = None,
    Metallic_Taste_in_Mouth: Optional[bool] = None,
    Dizziness_Trouble_Concentrating: Optional[bool] = None,
    Pain_in_Back_or_Sides: Optional[bool] = None,
    Nausea_Vomiting: Optional[bool] = None,

    Model1=Depends(AKI_model),
    Model2=Depends(Nephrotic_model),
    Model3=Depends(CKD_model),
):
    if (
        AKIDiagnosis is not None and
        InitialCreatinine is not None and
        PeakCreatinine is not None and
        UrineOutput is not None and
        Swelling is not None
    ):
        patient_data = {
            "AKIDiagnosis" :AKIDiagnosis,
            "InitialCreatinine" :InitialCreatinine,
            "PeakCreatinine" :PeakCreatinine,
            "UrineOutput" :UrineOutput,
            "Swelling" :Swelling,
        }
        results = Model_1(patient_data,Model1)
        return results
    elif (
        Proteinuria is not None and
        Edema is not None and
        Albumin_Level is not None
    ):  
        patient_data = {
            'Age':Age,
            'Gender':Gender,
            'Proteinuria':Proteinuria,
            'Edema':Edema,
            'Albumin_Level':Albumin_Level,
        }      
        results = Model_2(patient_data,Model2)
        return results
    elif (
        Change_in_Urination is not None or
        Swelling is not None or
        Metallic_Taste_in_Mouth is not None or
        Dizziness_Trouble_Concentrating is not None or
        Pain_in_Back_or_Sides is not None or
        Nausea_Vomiting is not None
    ):    
        patient_data = {
            'Change_in_Urination': Change_in_Urination,
            'Metallic_Taste_in_Mouth': Metallic_Taste_in_Mouth,
            'Dizziness_Trouble_Concentrating': Dizziness_Trouble_Concentrating,
            'Pain_in_Back_or_Sides': Pain_in_Back_or_Sides,
            'Nausea_Vomiting': Nausea_Vomiting,
        }
        results = Model_3(patient_data,Model3)
        return results
    else:
        return "Invalid input parameters"

def Model_1(patient_data,model):
    patient_data = pd.DataFrame([patient_data])
    AKIDiagnosis = {'Yes': 0, 'No': 1}
    Swelling = {'Yes': 0, 'No': 1}
    patient_data['AKIDiagnosis'] = patient_data['AKIDiagnosis'].map(AKIDiagnosis)
    patient_data['Swelling'] = patient_data['Swelling'].map(Swelling)
    results = model.predict(patient_data)
    if results==0:
        results = "Has AKI"
    else:
        results = "no AKI"
    return results

def Model_2(patient_data,model):
    patient_data = pd.DataFrame([patient_data])

    gender_mapping = {'Female': 0, 'Male': 1}

    proteinuria_mapping = {'Very High': 0, 'High': 1, 'Low': 2, 'Moderate': 3, 'Normal': 4}

    edema_mapping = {'Yes': 0, 'No': 1}

    albumin_mapping = {'Normal': 0, 'Low': 1, 'Very Low': 2}

    patient_data['Gender'] = patient_data['Gender'].map(gender_mapping)
    patient_data['Proteinuria'] = patient_data['Proteinuria'].map(proteinuria_mapping)
    patient_data['Edema'] = patient_data['Edema'].map(edema_mapping)
    patient_data['Albumin_Level'] = patient_data['Albumin_Level'].map(albumin_mapping)

    results = model.predict(patient_data)
    if results==0:
        results = "Has nephrotic syndrome"
    else:
        results = "no nephrotic syndrome"
    return results

def Model_3(patient_data,model):
    patient_data = pd.DataFrame([patient_data])
    results = model.predict(patient_data)[0]
    if results==0:
        results = "Has CKD"
    else:
        results = "no CKD"
    return results

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1",port=8000, log_level="info")