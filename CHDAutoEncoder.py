import pandas as pd
import numpy as np
dataset = pd.read_csv("framingham_heart_disease.csv")

X = dataset.iloc[:,:15]
Y = dataset.iloc[:,15:16]
X = X.drop(columns = ['currentSmoker', "education"])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'most_frequent')
X.iloc[:,1:14] = imputer.fit_transform(X.iloc[:,1:14])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras import regularizers
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard

'''
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))
# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 13, epochs = 4)
'''

nb_epoch = 500
batch_size = 128
input_dim = 13 #num of columns, 30
encoding_dim = 14
hidden_dim = int(encoding_dim / 2) #i.e. 7
learning_rate = 1e-7

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation='relu')(encoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)



autoencoder.compile(metrics=['accuracy'],loss='mean_squared_error',optimizer='adam')

cp = ModelCheckpoint(filepath="autoencoder_fraud.h5",save_best_only=True,verbose=0)

tb = TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=True)
autoencoder.fit(X_train, X_train,epochs=nb_epoch,batch_size=batch_size,shuffle=True,validation_data=(X_test, X_test),verbose=1,callbacks=[cp, tb])


X_test_predictions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_predictions, 2), axis=1)
#error_df = pd.DataFrame({'True_class': Y_test})
#error_df.describe()

X_train_predictions = autoencoder.predict(X_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_predictions, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=30,random_state=None)
classifier.fit(X_train_predictions, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,f1_score
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
acc_train=accuracy_score(y_train,classifier.predict(X_train_predictions))
cm_train = confusion_matrix(y_train,classifier.predict(X_train_predictions))
print(classification_report(y_train,classifier.predict(X_train_predictions)))
Y_pred = classifier.predict(X_test)
cm_test = confusion_matrix(y_test, Y_pred)
acc_test=accuracy_score(y_test, Y_pred)
print(classification_report(y_test, Y_pred))