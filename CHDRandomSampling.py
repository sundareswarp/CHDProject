import pandas as pd
import numpy as np
dataset = pd.read_csv("framingham_heart_disease.csv")
import matplotlib.pyplot as plt
X = dataset.iloc[:,:15]
y = dataset.iloc[:,15:16]
X = X.drop(columns = ['education', "currentSmoker"])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
X.iloc[:,1:14] = imputer.fit_transform(X.iloc[:,1:14])

#from imblearn.under_sampling import RandomUnderSampler
#rus = RandomUnderSampler(return_indices=True)
#X_rus, y_rus, id_rus = rus.fit_sample(X, y)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_sample(X, y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size = 0.2, random_state = None)

#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression(random_state = 0)
#classifier.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=30,random_state=None)
classifier.fit(X_train, y_train)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 10, algorithm = 'auto')
#classifier.fit(X_train,y_train)

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(random_state = 0)
#classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,f1_score
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
acc_train=accuracy_score(y_train,classifier.predict(X_train)) * 100
cm_train = confusion_matrix(y_train,classifier.predict(X_train))
print("Classification Report TRAIN : ")
print(classification_report(y_train,classifier.predict(X_train)))
y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_test, y_pred)
acc_test=accuracy_score(y_test, y_pred) * 100
print("Classification Report TEST : ")
print(classification_report(y_test, y_pred))
