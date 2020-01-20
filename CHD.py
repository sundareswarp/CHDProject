import pandas as pd
import numpy as np
dataset = pd.read_csv("framingham_heart_disease.csv")
#dataset = dataset.dropna()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile, f_classif

X = dataset.iloc[:,:15]
Y = dataset.iloc[:,15:16]
X = X.drop(columns = ['education', "currentSmoker"])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
X.iloc[:,1:14] = imputer.fit_transform(X.iloc[:,1:14])


'''Feature Selection 1'''
bestfeatures = SelectKBest(score_func=f_classif, k='all')
fit = bestfeatures.fit(X,Y)
scores1 = pd.DataFrame(fit.scores_)
scores1.to_csv('scorereport1.csv',index=False) 


'''Feature Selection 2'''
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,Y)
scores2 = model.feature_importances_
scores2 = pd.DataFrame(scores2)
scores2.to_csv('fs2.csv',index=False)

'''Feature Selection 3'''
import sklearn.feature_selection
f=sklearn.feature_selection.mutual_info_classif(X, Y, discrete_features='auto', n_neighbors=3, copy=True, random_state=None)
scores3 =pd.DataFrame(f)
scores3.to_csv('fs3.csv',index=False)



#X = X.drop(columns = [ 'education', 'heartRate', 'cigsPerDay'])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
'''

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0)
classifier.fit(X_train, Y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,f1_score
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
acc_train=accuracy_score(Y_train,classifier.predict(X_train))
cm_train = confusion_matrix(Y_train,classifier.predict(X_train))
print(classification_report(Y_train,classifier.predict(X_train)))
Y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(Y_test, Y_pred)
acc_test=accuracy_score(Y_test, Y_pred)
print(classification_report(Y_test, Y_pred))