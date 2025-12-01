import pandas as pd
from sklearn import metrics

veriler = pd.read_excel('C:/Users/salih/OneDrive/Masaüstü/pythonogren/Iris.xls')

x = veriler.iloc[:,0:4].values
y = veriler.iloc[:,4]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,y_train)

y_pred = LR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm,"Logistic reg")

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred2 = knn.predict(X_test)

cm2 = confusion_matrix(y_test, y_pred2)
print(cm2,"Knn classifier")

from sklearn.svm import SVC
svc = SVC(kernel ='linear')
svc.fit(X_train, y_train)

y_pred3 = svc.predict(X_test)
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3,"SVC")

from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(X_train, y_train)
y_pred4 = GNB.predict(X_test)
cm4 = confusion_matrix(y_test, y_pred4)
print(cm4,"GAUssian")

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')

dtc.fit(X_train,y_train)
y_pred5 = dtc.predict(X_test)

cm5 = confusion_matrix(y_test, y_pred5)

print(cm5,"DesicionTree")
#sonucu her seferimnde farklı döndürüyor
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1)
rfc.fit(X_train,y_train)
y_pred6 = rfc.predict(X_test)
y_proba1 = rfc.predict_proba(X_test)

cm6 = confusion_matrix(y_test, y_pred6)

print(cm6,"RandomForest ")

print(y_proba1[:,0])
fpr, tpr, thold = metrics.roc_curve(y_test, y_proba1[:,0], pos_label ='e') 
print(fpr,"burdayim")
print(tpr)


from sklearn.metrics import accuracy_score

print("LR accuracy:", accuracy_score(y_test, y_pred))
print("KNN accuracy:", accuracy_score(y_test, y_pred2))
print("SVC accuracy:", accuracy_score(y_test, y_pred3))
print("GNB accuracy:", accuracy_score(y_test, y_pred4))
print("Decision Tree accuracy:", accuracy_score(y_test, y_pred5))
print("Random Forest accuracy:", accuracy_score(y_test, y_pred6))
