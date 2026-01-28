https://risk-api-service-1769105645.azurewebsites.net/docs

Machine Learning Risk Classifier (REST API)
Tämä projekti on koneoppimiseen perustuva sovellus, joka luokittelee syöpäriskiä (pahanlaatuinen vs. hyvänlaatuinen). Sovellus on toteutettu REST API -rajapintana, mikä tarkoittaa, että mallia voidaan käyttää verkon yli lähettämällä sille tietoja.

Käytetty data
Mallin opettamiseen käytettiin Scikit-learn-kirjaston valmista Breast Cancer Wisconsin -datasettiä. Tämä on klassinen datasetti, joka sisältää tietoja solujen ominaisuuksista (kuten koko ja muoto), joiden perusteella tehdään ennuste.

Mallin kehitysvaiheet
Projektissa testattiin kahta eri lähestymistapaa ennusteen tekemiseen:

Logistinen regressio (Logistic Regression): Ensimmäinen versio perustui tähän yksinkertaiseen malliin. Se yrittää löytää suoran viivan, joka erottaa eri luokat toisistaan. Matemaattisesti se käyttää sigmoid-funktiota:$$P(y=1|x) = \frac{1}{1 + e^{-z}}$$

Random Forest (Nykyinen versio): Päivitin mallin käyttämään Random Forest -luokittelijaa, koska se on tarkempi. Se perustuu useiden eri päätöspuiden (Decision Trees) yhteistyöhön. Yksi puu voi erehtyä, mutta kun kymmenet puut "äänestävät" lopputuloksesta, ennusteesta tulee luotettavampi.

Projektin workflow
Projekti etenee loogisessa järjestyksessä datan käsittelystä pilveen:
Datan lataus: Haetaan breast_cancer -data sklearn-kirjastosta.
Koulutus (train.py): Malli opetetaan tunnistamaan datasta riippuvuudet ja tallennetaan .joblib-tiedostoksi.
Rajapinta (main.py): Luodaan FastAPI-palvelin, joka lataa tallennetun mallin ja vastaa kyselyihin.
Kontitus: Pakataan sovellus Docker-imagelle, jotta se toimii samalla tavalla Azuressa kuin omalla koneella.

Paikallinen käyttöohje
Voit ajaa projektin omalla koneellasi seuraavasti:
1. Asennus
Luo virtuaaliympäristö ja asenna tarvittavat kirjastot:
Bashpython3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
4. Mallin opettaminen
   Aja koodi, joka luo koneoppimismallin:
   Bashpython train.py
7. API-palvelimen käynnistys
8. Käynnistä REST API -palvelin:
9. Bash uvicorn main:app --reload
    
Sovellus on nyt käynnissä osoitteessa http://127.0.0.1:8000. 
Testidokumentaation löydät osoitteesta http://127.0.0.1:8000/docs.
4. Käyttö Dockerilla
Jos haluat testata sovellusta kontissa:
Bash docker build -t ml-risk-classifier .
docker run -p 8000:8000 ml-risk-classifier


