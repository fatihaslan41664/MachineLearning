import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/odev_tenis.csv')
weather = veriler.iloc[:,0:1].values
play = veriler.iloc[:,-1].values.reshape(-1,1)
windy= veriler.iloc[:,3].values.reshape(-1,1)
other_columns = veriler.iloc[:,1:3].values

ohe = OneHotEncoder()
weather = ohe.fit_transform(weather).toarray()
play = ohe.fit_transform(play).toarray()
play = play[:, 0]
windy = ohe.fit_transform(windy).toarray()
windy = windy[:,0]


weather_column = pd.DataFrame(data=weather, index = range(14), columns = ['overcast','rainy','sunny'])
playColumn = pd.DataFrame(data=play, index = range(14), columns = ['playable?'])
windyColumn = pd.DataFrame(data=play, index = range(14), columns = ['is_not_windy?'])
otherColumn = pd.DataFrame(data=other_columns, index = range(14), columns = ['temparature?','humidity'])

s1 = pd.concat([weather_column,otherColumn],axis=1)
s2 = pd.concat([s1,windyColumn],axis=1)
s3 = pd.concat([s1,playColumn],axis=1)

xdegisken = s3.drop(columns=['humidity'])
ydegisken = s3.iloc[:,4]

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int),values=xdegisken, axis=1)
X_List = xdegisken.iloc[:,[0,1,2,3,4]]
X_List=np.array(X_List, dtype=float)
model = sm.OLS(ydegisken,X_List).fit()
print(model.summary())

from sklearn.model_selection import train_test_split
#makina eğitiyorum yüzde 67 si ile eğitim yapıyor kalanı ile kendini doğruluyor
x_train,x_test,y_train,y_test = train_test_split(X_List, ydegisken,test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)