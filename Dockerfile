# 1. Valitaan pohja: Käytetään virallista Python 3.9 -versiota
FROM python:3.9-slim

# 2. Työkansio: Luodaan kontin sisälle kansio /app
WORKDIR /app

# 3. Kopioidaan requirements.txt kontin sisälle
COPY requirements.txt .

# 4. Asennetaan kirjastot kontin sisällä
RUN pip install --no-cache-dir -r requirements.txt

# 5. Kopioidaan kaikki loput tiedostot (main.py, mallit, yms.) konttiin
COPY . .

# 6. Avataan portti 80 (Azure tykkää portista 80)
EXPOSE 80

# 7. Käynnistyskomento: Kun kontti käynnistyy, aja tämä
# --host 0.0.0.0 tarkoittaa "kuuntele kaikkea liikennettä"
# --port 80 tarkoittaa "kuuntele porttia 80"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]