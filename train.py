import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

def train_professional_model():
    # --- 1. DATAN VALMISTELU ---
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Lisätään esimerkin vuoksi kategorinen sarake, jotta opit ColumnTransformerin
    # Oletetaan, että meillä on 'priority' -sarake (tämä on vain opetustarkoitukseen)
    X['priority'] = ['low', 'medium', 'high'] * 189 + ['low'] * 2 

    # Määritellään, mitkä sarakkeet ovat mitäkin tyyppiä
    numeric_features = data.feature_names.tolist()
    categorical_features = ['priority']

    # --- 2. ESIKÄSITTELYPUTKET (Pre-processing) ---
    # Numeroille: skaalaus (StandardScaler)
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Kategorioille: One-Hot Encoding (muuttaa tekstin nolliksi ja ykkösiksi)
    # handle_unknown='ignore' on tärkeä tuotannossa: jos malli kohtaa uuden kategorian, se ei kaadu.
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # --- 3. COLUMN TRANSFORMER (Yhdistäjä) ---
    # Tässä kerrotaan, mihin sarakkeisiin mikäkin muunnos kohdistuu.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # --- 4. LOPULLINEN PIPELINE ---
    # Nyt yhdistetään esikäsittely ja itse algoritmi.
    # RandomForestClassifier on "Päivä 2" -tason algoritmi: se on puupohjainen ja tehokas.
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # --- 5. KOULUTUS ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("⚙️ Koulutetaan ammattilaistason Pipelinea...")
    model_pipeline.fit(X_train, y_train)

    # --- 6. TALLENNUS ---
    # Tallenna malli. Tämä tiedosto sisältää nyt: 
    # 1. Skaalauksen säännöt, 2. Kategorioiden muunnokset, 3. Itse mallin.
    joblib.dump(model_pipeline, 'pro_model.joblib')
    print(" Malli tallennettu tiedostoon: pro_model.joblib")

if __name__ == "__main__":
    train_professional_model()