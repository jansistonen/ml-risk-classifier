import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

#Käytit nyt Logistista Regressiota. Kokeile vaihtaa Pipelineen jokin muu malli:
#RandomForestClassifier (erittäin suosittu ja tehokas "all-around" malli).
#SVC (Support Vector Classifier).
#Huomaat, että Scikit-learnin ansiosta koodista tarvitsee vaihtaa vain yksi rivi!

# Tuodaan oma moduuli
from data_loader import load_data

def train():
    # 1. Lataa data
    X, y = load_data()

    # 2. Jaetaan data (Train / Test split)
    # Stratify=y varmistaa, että luokkien suhde pysyy samana molemmissa seteissä
    # test_size=0.2 tarkoittaa, että 20% datasta jätetään sivuun testausta varten
    # random_state=42 varmistaa, että jako on aina sama (eli seed on sama, helpompi debugata)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. tehdään pipeline
    # Pipeline varmistaa, että skaalaus tehdään oikein (ei data leakagea)
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),          # Skaalaa datan (keskiarvo 0, varianssi 1)
        ('model', LogisticRegression())   # Itse malli
    ])

    # 4. Kouluta malli
    print("Koulutetaan mallia...")
    model_pipeline.fit(X_train, y_train)

    # 5. Testaa malli (predict)
    print("Testataan mallia")
    predictions = model_pipeline.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, predictions))

    # 6. Tallenna malli
    model_filename = 'risk_model.joblib'
    joblib.dump(model_pipeline, model_filename)
    print(f"malli tallennettu tiedostoon: {model_filename}")

if __name__ == "__main__":
    train()