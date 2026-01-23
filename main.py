# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Määritellään millaista dataa API ottaa vastaan (Schema)
# Oikeassa tuotannossa tässä lueteltaisiin kaikki 30 syöpädatan saraketta.
# Oikaisemme nyt hieman: määrittelemme vain muutaman, ja täytämme loput nollilla taustalla.
class PatientData(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    # ... kuvitellaan tähän loput 27 numeerista saraketta ...
    
    # --- TEHTÄVÄ 1: ---
    # Lisää tähän alle kenttä nimeltä 'priority'. 
    # Sen tyyppi pitää olla str (merkkijono), koska se on kategorinen (Low/Medium/High).
    # Kirjoita rivi tähän:
    priority: str
    


# 2. Ladataan koulutettu malli muistiin
# Tämä tehdään heti alussa, jotta mallia ei tarvitse ladata joka kerta kun joku kutsuu APIa.
try:
    model = joblib.load('pro_model.joblib')
    print("✅ Malli ladattu onnistuneesti.")
except FileNotFoundError:
    print("❌ Virhe: Mallitiedostoa ei löydy. Aja train.py ensin.")

# 3. Luodaan sovellus
app = FastAPI(title="Breast Cancer Risk API")

# 4. Luodaan endpoint (reitti)
@app.post("/predict")
def predict_risk(input_data: PatientData):
    # Tässä funktiossa tapahtuu taika.
    # input_data on nyt Pydantic-olio, joka sisältää lähettämäsi tiedot.
    
    # --- TEHTÄVÄ 2 (TÄRKEÄ): ---
    # Scikit-learnin pipeline odottaa Pandas DataFramea, jossa on sarakkeiden nimet.
    # Pydantic-olio pitää muuttaa sanakirjaksi (dict) ja sitten DataFrameksi.
    
    # 1. Muuta input_data sanakirjaksi. Vinkki: input_data.dict()
    data_as_dict = input_data.dict()   # <--- KIRJOITA KOODI TÄHÄN
    
    # 2. Luodaan DataFrame. Koska kyseessä on yksi potilas, se on yksi rivi.
    # Meidän pitää laittaa sanakirja listan sisään: [data_as_dict]
    input_df = pd.DataFrame([data_as_dict])
    
    # --- HACK (Koska emme kirjoittaneet kaikkia 30 saraketta) ---
    # Täytämme puuttuvat sarakkeet nollilla, jotta malli ei kaadu sarakkeiden puutteeseen.
    # Oikeassa työssä tätä EI tehdä, vaan Input-luokassa olisi kaikki kentät.
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0.0 # Täytetään puuttuva mittaus nollalla
            
    # Järjestetään sarakkeet samaan järjestykseen kuin koulutuksessa
    input_df = input_df[expected_columns]

    # --- TEHTÄVÄ 3: ENNUSTUS ---
    # Käytä ladattua 'model' -objektia ja sen .predict() -metodia.
    # Syötä sille input_df.
    prediction = model.predict(input_df)   # <--- KIRJOITA KOODI TÄHÄN
    
    # Haetaan todennäköisyys (vapaaehtoinen, mutta suositeltava)
    probability = model.predict_proba(input_df)[0].max()

    # Palautetaan vastaus JSON-muodossa
    return {
        "risk_class": "Malignant" if prediction[0] == 0 else "Benign", # Tarkista kummin päin luokat olivat datassasi!
        "probability": round(probability, 4),
        "input_summary": {
            "radius": input_data.mean_radius,
            "priority": input_data.priority
        }
    }