# -*- coding: utf-8 -*-
"""
Created on Tuesday, December 27 05:16:29 2022

@author: omerburakozgur
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import tensorflow as tf

model = tf.keras.models.load_model("D:\AIProjects\CNNModels\Model1HLD\Model1HLD.h5")
print("img")

"""
# Verisetimizi okuyoruz ve oluşturduğumuz dataframe'imize aktarıyoruz.
diabetes = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50])
# noDiabetes = np.array([1, 85, 66, 29, 0, 26.6, 0.351, 31])
sc = StandardScaler()
scaler = MinMaxScaler()
# Dosyadan verilerimizi okuyup dataframe'e aktarıyoruz
data = pd.read_csv("data//diabetes.csv")
df = pd.DataFrame(data)
# Okuduğumuz CSV dosyasından sütunları çekiyoruz ve daha kolay manipüle etmek için numpy array'e çeviriyoruz.
X = np.array(df.iloc[:, 0:8])
y = np.array(df.iloc[:, 8:9])

# Holdout metoduyla test train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Doğruluk oranını yükseltmek ve eğitim süresini kısaltmak için verilerimizi min max normalizasyon ile normalize ediyoruz
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MLP modelimiz (Multi Layer Perceptron)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=8, activation="relu"))
model.add(tf.keras.layers.Dense(units=64, activation="relu"))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(units=256, activation="relu"))
model.add(tf.keras.layers.Dense(units=128, activation="relu"))
model.add(tf.keras.layers.Dense(units=64, activation="relu"))
model.add(tf.keras.layers.Dense(units=32, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.build(input_shape=(32, 8))
# Model Özeti
print(model.summary())
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
# En doğru ağırlıkları tutmak ve modeli kaydetmek için model checkpoint kullandım
modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('bestmodel.hdf5', monitor='loss', save_best_only=True)
# Modelimizi Eğitiyoruz
history = model.fit(X_train, y_train, batch_size=32, epochs=100, callbacks=[modelCheckpoint])

# X-Test üzerinde tahmin yaptırıyoruz
predictions = model.predict(X_test)
#Tahmin Edilen float değerler
print(predictions)
# Confusion matrix'te label'ları kıyaslayabilmek için integer'a yuvarlamamız gerekiyor
predictions = (np.rint(predictions)).astype(int)
#Tahmin edilen ondalıklı değerlerin en yakın oldukları int değere yuvarlanmış halleri
print(predictions)

# Tekli tahmin için

diabetes = np.reshape(diabetes, (1, -1))
diabetes = scaler.transform(diabetes)
predictions = model.predict(diabetes)
print(predictions)
predictions = np.round(predictions)
print(predictions)

#Karışıklık Matrisi
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
disp.plot()
plt.show()
# Tahmin doğruluk değeri ortalama 96%
accuracy = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
print(accuracy)

#Loss grafiğinde minimum loss'u belli etmek için işaretliyoruz
minLoss = np.min(history.history['loss'])
lossArray = np.array(history.history['loss'])
lossPoint = np.where(lossArray == minLoss)
lossPoint = lossPoint[0] + 1

#Loss - Epoch Grafiği
plt.figure(0)
plt.plot(history.history["loss"], label="Training Loss")
plt.title("Epoch - Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.scatter(x=lossPoint, y=minLoss, edgecolors='red')
plt.legend()

#Accuracy - Epoch Grafiği
plt.figure(1)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.title("Epoch - Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""

