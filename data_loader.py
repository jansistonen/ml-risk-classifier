import pandas as pd
from sklearn.datasets import load_breast_cancer

def load_data():
    """
    Lataa datasetin ja palauttaa ominaisuudet (X) ja kohdemuuttujan (y).
    Simuloidaan 'Risk Classification' -skenaariota.
    """
    # Ladataan data (Breast Cancer dataset on klassinen binääriluokittelu)
    data = load_breast_cancer()
    
    # Muutetaan DataFrameksi selkeyden vuoksi
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"Data ladattu: {X.shape[0]} riviä, {X.shape[1]} ominaisuutta.")
    return X, y

if __name__ == "__main__":
    # Testataan, että lataus toimii
    X, y = load_data()
    print(X.head())