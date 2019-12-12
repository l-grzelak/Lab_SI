from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
# Wczytaj przykładowy zbiór danych - dane dotyczące trzech gatunków Irysów
iris = datasets.load_iris()
# Zadanie 1: sprawdź poniżej inne elementy wczytanego zbioru danych, w szczególności opis.
# Opisz w max 3 zdaniach swoimi słowami co zawiera zbiór danych
print('Opis irysów w zbiorze: ', iris['DESCR'])
print('\n Fature_names zbioru: ', iris['feature_names'])
print('\n nazwa pliku', iris['filename'])

#Zbiór danych zawiera 150 instancji po 50 z każdej z 3 klas
# Każda ma 4 atrybuty + klase
#- długośc i szerokość pręcika
#- długośc i szerokość platków
#Podsumowanie
#Bibliografie

# Zadanie 2:
# Stwórz listę kilku wybranych przez siebie wartości dla parametru n_neighbors
# W pętli 'for' użyj kolejnych wartości parametru do stworzenia klasyfikatora
# Następnie naucz go na danych uczących
# Zapisz wynik scoringu na danych testowych do osobnej listy




#Definiowanie wybranych wartosci parametru n_neighbors
lista_n = [1, 2, 5, 7, 13, 15, 19] 
#stworzenie listy do wpisywania wyników scoringu
dokladnosci = []

#podział na etykiety
X = iris.data
y = iris.target

# podział na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

for n_neighb in lista_n:
    # tworzenie klasyfikatora
    knn = KNeighborsClassifier(n_neighbors = n_neighb)
    #uczenie klasyfikatora
    knn.fit(X_train, y_train)
    #predykcja wartości dla zbioru testowego
    y_pred = knn.predict(X_test)
    #wykonanie scoringu 
    dokladnosc = knn.score(X_test, y_test)
    #wpisanie wyniku scoringu do listy
    dokladnosci.append(dokladnosc)
# Wyświetlanie wykresu zależności między liczbą sąsiadów a dokładnością.
print("Wykres dokładnosci")
#%matplotlib inline
plt.bar(lista_n, dokladnosci)
plt.xlabel("Liczba sąsiadów")
plt.ylabel("Dokładność scoringu")
plt.show()

# wczytaj dane o winach za pomocą funkcji poniżej


# Zbadaj zbiór danych. Stwórz wykresy obrazujące ten zbiór danych.
# Podziel zbiór danych na uczący i testowy.
# Wytrenuj klasyfikator kNN
# Dokonaj predykcji na zbiorze testowym
# Wypisz raport z uczenia: confusion_matrix oraz classification_report
#Wczytanie zbioru danych 
wine = datasets.load_wine()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%matplotlib inline
import seaborn as sns

# Konwersja na obiekt pandas.DataFrame
wine_df = pd.DataFrame(wine['data'], columns=wine['feature_names'])

# Konwersja warości liczbowych na opis gatunku
targets = map(lambda x: wine['target_names'][x], wine['target'] )

# doklejenie informacji o gatunku do reszty dataframe
wine_df['species'] = np.array(list(targets))

# rysowanie wykresu obrazującego zbiór danych 
sns.pairplot(wine_df, hue='species')
plt.show()


#podział na etykiety
X = wine.data
y = wine.target

#podział zbioru na zbiór uczący i zbiór testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# Tworzenie klasyfikatora knn
knn = KNeighborsClassifier(n_neighbors = 5)

# Trening klasyfikatora
knn.fit(X_train, y_train)

# przewidywanie wartoci dla zbioru testowego
y_pred = knn.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

# Raport z uczenia 
print("raport confusion_matrix")
print(confusion_matrix(y_test, y_pred))
print("raport classification_report")
print(classification_report(y_test, y_pred))