#maaslar polinomal regresyon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/maaslar.csv')

egitim_seviyesi = veriler.iloc[:,1].values.reshape(-1,1)
maaslar = veriler.iloc[:,2].values.reshape(-1,1)

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
"""plt.scatter(egitim_seviyesi, maaslar, color='red')
plt.plot(egitim_seviyesi, lin_reg2.predict(egitimseviye_poly_for_plot), color='blue')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')"""

X_new = [[20]]
X_new_poly = regresyon.transform(X_new)  # sadece transform, fit_transform değil
tahmin = lin_reg2.predict(X_new_poly)
#print(tahmin)

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
sc2 = StandardScaler()
egitimOlcek = sc1.fit_transform(egitim_seviyesi)
maasOlcek = sc2.fit_transform(maaslar).ravel() 

from sklearn.svm import SVR

svr_Reg = SVR(kernel='rbf')
svr_Reg.fit(egitimOlcek,maasOlcek)

# SVR epsilon tüpünü görselleştirmek
epsilon = svr_Reg.epsilon
y_pred = svr_Reg.predict(egitimOlcek)


"""plt.scatter(egitimOlcek, maasOlcek, color='red')
plt.plot(egitimOlcek, y_pred, color='blue', label='SVR Tahmin')

# epsilon bandı
plt.plot(egitimOlcek, y_pred + epsilon, 'g--', label='Epsilon Tüpü')
plt.plot(egitimOlcek, y_pred - epsilon, 'g--')

plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.legend()
plt.show()"""
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(egitim_seviyesi, maaslar)
"""plt.scatter(egitim_seviyesi, maaslar, color='red')
plt.plot(egitim_seviyesi, r_dt.predict(egitim_seviyesi), color = 'blue')
print(r_dt.predict([[7]]))"""

from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators=10, random_state=0)
RFR.fit(egitim_seviyesi, maaslar.ravel())
plt.scatter(egitim_seviyesi, maaslar, color='red')
plt.plot(egitim_seviyesi, RFR.predict(egitim_seviyesi), color = 'blue')
print(RFR.predict([[6.6]]))
from sklearn.metrics import r2_score
y_pred_scaled = svr_Reg.predict(egitimOlcek)
print("al tsatır")
print(r2_score(maaslar, RFR.predict(egitim_seviyesi)))
print("SVR R² (scaled):", r2_score(maasOlcek, y_pred_scaled))





































