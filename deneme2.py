import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/veriler.csv')

cinsiyet = veriler.iloc[:,4:5].values
ulke = veriler.iloc[:,0:1]

#le değişkenim bir encoder (kod kırıcı) oldu
le = preprocessing.LabelEncoder()
ohe = OneHotEncoder()
#ülke değişkenim şu anda kırıldldı 001 100 010 şeklinde ama değişkende gösteremiyorum 1 2 0 şeklinde gösteriliyor 1 in bulunduğu indexe göre
cinsiyet_ohe = ohe.fit_transform(cinsiyet).toarray()

ulke_ohe = ohe.fit_transform(ulke).toarray()

sonuc0 = pd.DataFrame(data=ulke_ohe, index = range(22), columns = ['fr','tr','us'])

diger_veri = veriler.iloc[:,1:4].values
sonuc1 = pd.DataFrame(data=diger_veri, index = range(22), columns = ['boy','kilo','yas'])
sonuc2 = pd.DataFrame(data=cinsiyet_ohe[:,0:1], index = range(22), columns = ['erkek_mi?'])
s1 = pd.concat([sonuc1,sonuc2],axis=1)
s2= pd.concat([sonuc0,s1],axis=1)
deneme1 = s2.drop(columns=['boy']) # son sütun hariç hepsi
deneme2 = s2.iloc[:, 3:4]

#önce 4 te ekliydi onu silince daha düzgün bir p value elde ettim yani backword elimination yaptım
import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values=deneme1, axis=1)
X_List = deneme1.iloc[:,[0,1,2,3,5]]
X_List=np.array(X_List, dtype=float)
model = sm.OLS(deneme2,X_List).fit()
print(model.summary())
from sklearn.model_selection import train_test_split
#makina eğitiyorum yüzde 67 si ile eğitim yapıyor kalanı ile kendini doğruluyor
x_train,x_test,y_train,y_test = train_test_split(X_List, deneme2,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)