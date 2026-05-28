import xgboost as xgb
import pandas as pd
import numpy as np


try:
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    print(" Model załadowany poprawnie!")
except Exception as e:
    print(f" Błąd wczytywania modelu: {e}")
    exit()


klasy = {0: "GALAXY", 1: "QSO", 2: "STAR"}

print("\n--- KLASYFIKATOR OBIEKTÓW KOSMICZNYCH ---")
print("Wpisz dane, aby otrzymać wynik.\n")

try:
    u = float(input("Podaj u: "))
    g = float(input("Podaj g: "))
    r = float(input("Podaj r: "))
    i = float(input("Podaj i: "))
    z = float(input("Podaj z: "))
    redshift = float(input("Podaj redshift (np. 0.5): "))
    alpha = float(input("Podaj alpha: "))
    delta = float(input("Podaj delta: "))


    input_data = pd.DataFrame([[u, g, r, i, z, redshift, alpha, delta]], 
                              columns=['u', 'g', 'r', 'i', 'z', 'redshift', 'alpha', 'delta'])

  
    predykcja = model.predict(input_data)[0]
    wynik = klasy[predykcja]

    print("\n" + "="*30)
    print(f"WYNIK: {wynik}")
    print("="*30)

except ValueError:
    print(" Błąd: Musisz wpisać liczby!")
except Exception as e:
    print(f" Wystąpił błąd: {e}")