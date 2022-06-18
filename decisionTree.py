import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.model_selection import cross_val_score

#gender - płeć (0,1-kobieta,mężczyzna), age - wiek (w latach), height - wzrost (w metrach)
#family_history_with_overweight - otyłość w rodzinie (nie/tak), FAVC - częste spożycie kalorycznych posiłków (nie/tak)
#FCVC - częstość jedzenia warzyw (skala 0-3), NCP - , CAEC-
#Smoke - palenie papierosów (nie/tak), CH2O - litry pitej wody, SCC - 
#FAF -, TUE -, CALC - ,
#MTRANS - najczęstszy środek transportu (0 - samochód, 1 - motor, 2 - rower, 3 - komunikacja miejska, 4 - pieszo)

#odczytanie pliku oraz podział kolumn na zawierające cechy i wynik
obesity = pd.read_csv("C:/Users/Krzys/Desktop/projekt_marcin/obesity1.csv", usecols=[i for i in range(0,17)], header=0)
kolumny_cech = ['Gender','Age','Height','Weight','family_history_with_overweight','FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS']
X = obesity[kolumny_cech]
Y = obesity.NObeyesdad

#podział zbiorów na trenujące i testowe
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#stworzenie drzewa, wytrenowanie go na zbiorze, a następnie sprawdzenie dokładności na zbiorze testowym
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#podzielenie danych na 5 podzbiorów testowych i sprawdzenie dokładności każdego z nich
scores = cross_val_score(clf,X,Y,cv=5)
print(scores)
print("algorytm zadziałał z dokładnością {0:.2f}{1} oraz odchyleniem standardowym równym {2:.3f}".format(scores.mean()*100,'%', scores.std()))
