import pandas as pd
import numpy as np
import seaborn as sns
#import matplotlib.pyplot as mp
dataset = pd.read_csv("framingham_heart_disease.csv")

X = dataset.iloc[:,:15]
y = dataset.iloc[:,15:16]
X = X.drop(columns = ['education', "currentSmoker"])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
X.iloc[:,1:14] = imputer.fit_transform(X.iloc[:,1:14])
