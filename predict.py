import joblib
import pandas as pd

# 1. Lataa tallennettu malli (Pipeline)
model = joblib.load('risk_model.joblib')

# 2. Luodaan "uusi" potilas (esimerkkiarvoja datasta)
# Huom: sarakkeiden määrän ja nimien on oltava samat kuin koulutuksessa
new_data = pd.DataFrame([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                          1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 
                          0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 
                          0.4601, 0.1189]], 
                        columns=model.feature_names_in_)

# 3. Tehdään ennuste
prediction = model.predict(new_data)
probability = model.predict_proba(new_data) # Todennäköisyys

print(f"Ennuste: {'Hyvänlaatuinen' if prediction[0] == 1 else 'Pahanlaatuinen'}")
print(f"Todennäköisyys: {probability.max():.2%}")