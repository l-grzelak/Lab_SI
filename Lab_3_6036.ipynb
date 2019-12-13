# Zadanie 1
# Wybierz inną cechę i spróbuj przewidzieć ceny mieszkań. Użyj walidacji krzyżowej.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

# Załadowanie zbioru nieruchomosci wraz z ich cenami 
mieszkania_boston = load_boston()
tax = mieszkania_boston['data'][:, np.newaxis, 9]
plt.scatter(tax, mieszkania_boston['target'])
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

# Stworzenie regresora liniowego
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(tax, mieszkania_boston['target'], test_size = 0.3)
linreg.fit(X_train, y_train)

# Przewidywanie ceny
y_pred = linreg.predict(X_test)

# domyślna metryka
print('Metryka domyślna: ', linreg.score(X_test, y_test))

# współczynniki regresji
print('Współczynniki regresji:\n', linreg.coef_)

#Utworzenie wykresu z przewidywanymi cenami mieszkań 
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='black', linewidth=3)
plt.show()
from sklearn.model_selection import cross_val_score
print("Walidacja krzyżowa")
cv_score_mse = cross_val_score(linreg, tax, mieszkania_boston.target, cv=5, scoring='neg_mean_squared_error')
print(cv_score_mse)