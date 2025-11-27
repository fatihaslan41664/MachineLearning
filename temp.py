import pandas as pd
import numpy as np
import math
from sklearn.impute import SimpleImputer

class Main:     
        def __init__(self):
            print("Main sınıfı oluşturuldu")



veriler = pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/veriler.csv')
eksikveriler=pd.read_csv('C:/Users/salih/OneDrive/Masaüstü/pythonogren/eksikveriler.csv')
yas = eksikveriler['yas']
"""
print(yas)
y = 0
z = 0
for x in yas:
        
    if not math.isnan(x):
        y = x+ y
        z = z+1
        print(z)
        
print (y/z)
print(y)
"""
"""boy kilo ve yaş sütünlarını ayırdım"""
Yas = eksikveriler.iloc[:, 1:4]  # 1,2,3 sütunları
"""ortalama stratejisini belilerdim bundan sonra imputer değişkenim eksik verilere ortalamalarıyla dolduran değişken oldu"""
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#yas_field değişkenim artık ortalama verilerle donatılmış bir değişken
Yas_filled = imputer.fit_transform(Yas)  # numpy array döner

print(Yas_filled)
# Kontrol
from sklearn import preprocessing
#ulkere sutunun tüm satırlarını çektim
ulke = eksikveriler.iloc[:,0:1]

#le değişkenim bir encoder (kod kırıcı) oldu
le = preprocessing.LabelEncoder()
#ülke değişkenim şu anda kırıldldı 001 100 010 şeklinde ama değişkende gösteremiyorum 1 2 0 şeklinde gösteriliyor 1 in bulunduğu indexe göre
ulke.iloc[:, 0] = le.fit_transform(ulke.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
#ulke değişkenimin içindeki veriler artık gösteriliyor 1 2 0 şekilde değil 001 010 100 ve bunları int e çevirdim artık double dönmüyor
ulke = ohe.fit_transform(ulke).toarray()
ulke = ulke.astype(int)
#şu anda array olan dizimi bir dataframeye çevirdim ileride bu dizileri birleştirmeme yarıyor hemde tablom okunaklı oluyor hem başlıkları var
# hem de index ataması yapılıyor
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
#aynı işlemi boy kilo yas olarak yapıyorum
sonuc2 = pd.DataFrame(data = Yas_filled, index = range(22), columns = ['boy','kilo','yas'])

#cinsiyet kolonunu seçtim
cinsiyet = eksikveriler.iloc[:,-1].values
#cinsiyet kolonunu bir data frame olarak yapılandırıyorum
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22),columns=["cinsiyet"])

#ulke ile yas,cinsiyet,boy dataframelerini birleştirdim
s1 = pd.concat([sonuc,sonuc2],axis=1)
# ve bunlara cinsiyetlerini ekledim

s2 = pd.concat([s1,sonuc3],axis=1)


from sklearn.model_selection import train_test_split
#makina eğitiyorum yüzde 67 si ile eğitim yapıyor kalanı ile kendini doğruluyor
x_train,x_test,y_train,y_test = train_test_split(s1, sonuc3,test_size=0.33,random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
xintrain = sc.fit_transform(x_train)
xintest = sc.fit_transform(x_test)











































































