#zadanie 1 
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


samochody = fetch_openml('cars1')
#0 - MPG (ile mil można przejechać na jedym galonie paliwa)
#4 weightLbs (waga w Lbs)
X = samochody.data[:,[0,4]]

# Konwersja typów dla y

y = samochody['target']
y = [int(elem) for elem in y]

#Podział na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
#Tworzenie klasyfikatora z 3 klastrami
kmn = KMeans(n_clusters=3)
#nauka klasyfikatora danych treningowych
kmn.fit(X_train)
centra = kmn.cluster_centers_
fig, ax = plt.subplots(1, 2)
ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)
y_pred_train = kmn.predict(X_train)
ax[1].scatter(X_train[:, 0], X_train[:, 1], c=y_pred_train, s=20)
ax[1].scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()
import matplotlib.pyplot as plt
y_pred = kmn.predict(X_test)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20)
plt.scatter(centra[:, 0], centra[:, 1], c='red', s=50)
plt.show()

#zadanie 2
#Na podstawie  wykresów można stwierdzić, że samochody w zbiorze maj ą różne masy całkowite. Najcięższe samochody to terenówki,lżejsze to kombi/limuzyny a najlżejsze to lekkie auta miejskie 