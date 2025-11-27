#maaslar polinomal regresyon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/maaslar.csv')

egitim_seviyesi = veriler.iloc[:,1].values.reshape(-1,1)
maaslar = veriler.iloc[:,2].values

from sklearn.linear_model import LinearRegression
#lineer
lin_reg = LinearRegression()
lin_reg.fit(egitim_seviyesi,maaslar)
"""plt.scatter(egitim_seviyesi, maaslar, color='red')
plt.plot(egitim_seviyesi, lin_reg.predict(egitim_seviyesi), color='blue')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')"""

from sklearn.preprocessing import PolynomialFeatures
regresyon = PolynomialFeatures(degree=7)
egitimseviye_poly = regresyon.fit_transform(egitim_seviyesi)

lin_reg2 = LinearRegression()
lin_reg2.fit(egitimseviye_poly, maaslar)

# Tahmin için:
egitimseviye_poly_for_plot = regresyon.transform(egitim_seviyesi)
plt.scatter(egitim_seviyesi, maaslar, color='red')
plt.plot(egitim_seviyesi, lin_reg2.predict(egitimseviye_poly_for_plot), color='blue')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')

X_new = [[20]]
X_new_poly = regresyon.transform(X_new)  # sadece transform, fit_transform değil
tahmin = lin_reg2.predict(X_new_poly)
print(tahmin)