import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
import sys
import joblib
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from makineOgrenmesiArayuz import Ui_MainWindow
from joblib import dump, load
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from sklearn import tree

# Kullanıcı girdisini al
# Kullanıcı girdisine göre predict at
# Yapılan şeyleri ve hata mesajlarını label'a yazdır.
# Belki modeller tekrardan eğitilebilir

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        # %% Değişkenler
        self.setupUi(self)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.df = None
        self.diabetes = np.array([[1, 85, 66, 29, 0, 26.6, 0.351, 31]])
        # diabetes = np.array([6, 148, 72, 35, 0, 33.6, 0.627, 50])
        # noDiabetes = np.array([1, 85, 66, 29, 0, 26.6, 0.351, 31])
        self.minLoss = None
        self.lossArray = None
        self.lossPoint = None
        self.predictions = None
        self.imageCM = "TempPlots/CM.png"
        self.imageLoss = "TempPlots/Loss.png"
        self.imageAccuracy = "TempPlots/Accuracy.png"
        QMainWindow.showMaximized(self)
        self.tabWidget.showMaximized()
        self.sc = StandardScaler()
        self.scaler = MinMaxScaler()

        self.headers = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI",
                        "Diabetes Pedigree Function",
                        "Age"]

        self.VeriSetiniYukle()
        self.VeriSetiniAyirHoldout()
        self.NormalizeEt()
        self.btnModelKullan.clicked.connect(self.ModelKullan)
        # self.btnModelEgit.clicked.connect(lambda: self.ModelEgit())
        self.btnModelEgit.clicked.connect(self.ModelEgit)

    def VeriSetiniYukle(self):
        # Verisetimizi okuyoruz ve oluşturduğumuz dataframe'imize aktarıyoruz.
        data = pd.read_csv("data//diabetes.csv")
        self.df = pd.DataFrame(data)
        self.X = np.array(self.df.iloc[:, 0:8])
        self.y = np.array(self.df.iloc[:, 8:9])

    def KullaniciGirdisiniAktar(self):
        self.diabetes[0][0] = float(self.txtHamilelik.text())
        self.diabetes[0][1] = float(self.txtKanSekeri.text())
        self.diabetes[0][2] = float(self.txtKanBasinci.text())
        self.diabetes[0][3] = float(self.txtDeri.text())
        self.diabetes[0][4] = float(self.txtInsulin.text())
        self.diabetes[0][5] = float(self.txtVucutKitle.text())
        self.diabetes[0][6] = float(self.txtGenetik.text())
        self.diabetes[0][7] = float(self.txtYas.text())
        print(self.diabetes)
        print(np.shape(self.diabetes))

    def NormalizeEt(self):
        # Sklearn kütüphanesinden Standard Scaler kullanarak özelliklerimizi -1 ve 1 aralığına çekiyoruz.
        self.VeriSetiniAyirHoldout()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)

    def VeriSetiniAyirHoldout(self):
        # Modelimizi eğitip test etmek için tek parça olan verisetimizi eğitim ve test olmak üzere ikiye ayırıyoruz.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.10,
                                                                                random_state=0)
        self.EgitimVerisetiniAktar(self.X_train)
        self.TestVerisetiniAktar(self.X_test)
        # self.EgitimVerisetiniAktar()
        # self.TestVerisetiniAktar()

    def ModelKullan(self):
        if self.cmbMimari.currentIndex() == 0:
            model = tf.keras.models.load_model("D:\AIProjects\Models\ANNH\ANNH.h5")
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNH\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNH\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNH\Accuracy.png"))
            print("0")

        elif self.cmbMimari.currentIndex() == 1:
            model = tf.keras.models.load_model("D:\AIProjects\Models\ANNK\ANNKF.h5")
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNK\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNK\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\ANNK\Accuracy.png"))
            print("1")


        elif self.cmbMimari.currentIndex() == 2:
            model = pickle.load(open('D:\AIProjects\Models\KNNH\KNNH.h5', 'rb'))
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNH\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNH\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNH\Accuracy.png"))
            print("2")


        elif self.cmbMimari.currentIndex() == 3:
            model = pickle.load(open('D:\AIProjects\Models\KNNK\KNNKF.h5', 'rb'))
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNK\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNK\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\KNNK\Accuracy.png"))
            print("3")


        elif self.cmbMimari.currentIndex() == 4:
            model = pickle.load(open('D:\AIProjects\Models\DTCH\DTCH.h5', 'rb'))
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCH\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCH\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCH\Accuracy.png"))
            print("4")


        elif self.cmbMimari.currentIndex() == 5:
            model = pickle.load(open('D:\AIProjects\Models\DTCK\DTCKF.h5', 'rb'))
            self.Predict(model)
            self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCK\CM.png"))
            self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCK\Loss.png"))
            self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\Models\DTCK\Accuracy.png"))
            print("5")

    def ModelEgit(self):

        if self.cmbMimari.currentIndex() == 0:
            print("ANN H")
            self.ModelANNHoldout()
        elif self.cmbMimari.currentIndex() == 1:
            print("ANN K")
            self.ModelANNKfold()
        elif self.cmbMimari.currentIndex() == 2:
            print("KNN H")
            self.ModelKNNHoldout()
        elif self.cmbMimari.currentIndex() == 3:
            print("KNN K")
            self.ModelKNNKfold()
        elif self.cmbMimari.currentIndex() == 4:
            print("DTC H")
            self.ModelDTCHoldout()
        elif self.cmbMimari.currentIndex() == 5:
            print("DTC K")
            self.ModelDTCKfold()

    def ModelANNHoldout(self):
        # YSA'nın hazırlanması
        self.VeriSetiniAyirHoldout()
        self.NormalizeEt()

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=8, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.build(input_shape=(32, 8))

        # Modelimizi derliyoruz. Çıktı katmanında softmax kullandığımız için loss fonksiyonu olarak da binary-crossentropy kullandık.
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        # Optimal Accuracy ve Loss değerlerini elde etmek için Model Checkpoint ve callback kullanıyoruz.
        modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('bestmodel.hdf5', monitor='loss', save_best_only=True)
        history = self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=100)

        minLoss = np.min(history.history['loss'])
        lossArray = np.array(history.history['loss'])
        lossPoint = np.where(lossArray == minLoss)
        lossPoint = lossPoint[0] + 1

        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.sc.transform(self.diabetes)
        singlePrediction = self.model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

        predictions = self.model.predict(self.X_test)
        predictions = (np.rint(predictions)).astype(int)

        self.minLoss = np.min(history.history['loss'])
        self.lossArray = np.array(history.history['loss'])
        self.lossPoint = np.where(self.lossArray == self.minLoss)

        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")
        plt.show()

        accuracy = accuracy_score(self.y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)

        plt.figure(0,figsize=(4.8,3.6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.title("Epoch - Loss Graph")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.scatter(x=lossPoint, y=minLoss, edgecolors='red')
        plt.legend()
        plt.savefig("TempPlots/Loss.png")


        plt.figure(1,figsize=(4.8,3.6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.title("Epoch - Accuracy Graph")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()
        self.model.save("ANNH.h5")
        self.GoruntuYukle()
        # self.GrafikOlustur(history)

    def ModelANNKfold(self):

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=8, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(units=256, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=128, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=64, activation="relu"))
        self.model.add(tf.keras.layers.Dense(units=32, activation="relu"))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.build(input_shape=(32, 8))
        print(self.model.summary())
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
        # modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('bestmodel.hdf5', monitor='loss', save_best_only=True)
        # history = self.model.fit(self.X_train, self.y_train, batch_size=32, epochs=100, callbacks=[modelCheckpoint])
        k = 5
        kfold = KFold(n_splits=k)
        for train_index, test_index in kfold.split(self.X):
            # Get the training and test data for this fold
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            self.EgitimVerisetiniAktar(X_train)
            self.TestVerisetiniAktar(X_test)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            # Train the model on the training data
            history = self.model.fit(X_train, y_train, batch_size=32, epochs=20)

        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.scaler.transform(self.diabetes)
        singlePrediction = self.model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

        predictions = self.model.predict(X_test)
        predictions = (np.rint(predictions)).astype(int)

        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")

        plt.show()
        accuracy = accuracy_score(y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)

        minLoss = np.min(history.history['loss'])
        lossArray = np.array(history.history['loss'])
        lossPoint = np.where(lossArray == minLoss)
        lossPoint = lossPoint[0] + 1

        plt.figure(0,figsize=(4.8,3.6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.title("Epoch - Loss Graph")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig("TempPlots/Loss.png")
        plt.scatter(x=lossPoint, y=minLoss, edgecolors='red')
        plt.legend()

        plt.figure(1,figsize=(4.8,3.6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.title("Epoch - Accuracy Graph")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()
        self.model.save("ANNKF.h5")

        self.GoruntuYukle()

    def ModelKNNHoldout(self):
        self.VeriSetiniAyirHoldout()
        self.NormalizeEt()

        self.model = KNeighborsClassifier(n_neighbors=2)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.model.fit(self.X_train, self.y_train)

        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.scaler.transform(self.diabetes)

        singlePrediction = self.model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

        predictions = self.model.predict(self.X_test)
        predictions = (np.rint(predictions)).astype(int)

        # LOSS======================
        # Generate the training and validation scores for each epoch
        plt.figure(0, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5,
                                                                scoring='neg_log_loss')
        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, -train_mean, label='Training Loss')
        plt.plot(train_sizes, -test_mean, label='Validation Loss')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Loss.png")
        plt.show()
        # ACCURACY======================

        # Generate the training and validation scores for each epoch
        plt.figure(1, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5)

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, train_mean, label='Training Score')
        plt.plot(train_sizes, test_mean, label='Validation Score')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()

        # CM======================
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")
        plt.show()
        accuracy = accuracy_score(self.y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)
        self.GoruntuYukle()
        pickle.dump(self.model, open('KNNH.h5', 'wb'))

    def ModelKNNKfold(self):
        self.model = KNeighborsClassifier(n_neighbors=2)

        k = 5
        kfold = KFold(n_splits=k)
        for train_index, test_index in kfold.split(self.X):
            # Get the training and test data for this fold
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.EgitimVerisetiniAktar(self.X_train)
            self.TestVerisetiniAktar(self.X_test)
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            # Train the model on the training data
            self.model.fit(self.X_train, self.y_train)
            # history = model.fit(X_train, y_train, batch_size=32, epochs=20)
            # Evaluate the model on the test data
            accuracy = self.model.score(self.X_test, self.y_test)
            print(f'Fold accuracy: {accuracy:.2f}')

        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.scaler.transform(self.diabetes)

        singlePrediction = self.model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

        predictions = self.model.predict(self.X_test)
        predictions = (np.rint(predictions)).astype(int)

        # LOSS======================
        # Generate the training and validation scores for each epoch
        plt.figure(0, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5,
                                                                scoring='neg_log_loss')

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, -train_mean, label='Training Loss')
        plt.plot(train_sizes, -test_mean, label='Validation Loss')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Loss.png")
        plt.show()

        # ACCURACY======================

        # Generate the training and validation scores for each epoch
        plt.figure(1, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5)

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, train_mean, label='Training Score')
        plt.plot(train_sizes, test_mean, label='Validation Score')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()

        # CM======================
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")
        plt.show()

        accuracy = accuracy_score(self.y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)
        pickle.dump(self.model, open('KNNKF.h5', 'wb'))
        self.GoruntuYukle()

    def ModelDTCHoldout(self):
        self.VeriSetiniAyirHoldout()
        self.NormalizeEt()
        self.model = tree.DecisionTreeClassifier()
        self.model.fit(self.X_train, self.y_train)

        predictions = self.model.predict(self.X_test)
        predictions = (np.rint(predictions)).astype(int)

        # LOSS======================
        # Generate the training and validation scores for each epoch
        plt.figure(0, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5,
                                                                scoring='neg_log_loss')

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, -train_mean, label='Training Loss')
        plt.plot(train_sizes, -test_mean, label='Validation Loss')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Loss.png")
        plt.show()

        # ACCURACY======================
        # Generate the training and validation scores for each epoch
        plt.figure(1, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5)

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, train_mean, label='Training Score')
        plt.plot(train_sizes, test_mean, label='Validation Score')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()

        # CM======================
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")
        plt.show()

        accuracy = accuracy_score(self.y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)
        pickle.dump(self.model, open('DTCH.h5', 'wb'))
        self.GoruntuYukle()

    def ModelDTCKfold(self):
        self.model = tree.DecisionTreeClassifier()

        k = 5
        kfold = KFold(n_splits=k)
        for train_index, test_index in kfold.split(self.X):
            # Get the training and test data for this fold
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.EgitimVerisetiniAktar(self.X_train)
            self.TestVerisetiniAktar(self.X_test)
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            # Train the model on the training data
            self.model.fit(self.X_train, self.y_train)
            # history = model.fit(X_train, y_train, batch_size=32, epochs=20)
            # Evaluate the model on the test data
            accuracy = self.model.score(self.X_test, self.y_test)
            print(f'Fold accuracy: {accuracy:.2f}')

        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.scaler.transform(self.diabetes)

        singlePrediction = self.model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

        predictions = self.model.predict(self.X_test)
        predictions = (np.rint(predictions)).astype(int)

        # LOSS======================
        # Generate the training and validation scores for each epoch
        plt.figure(0, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5,
                                                                scoring='neg_log_loss')

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, -train_mean, label='Training Loss')
        plt.plot(train_sizes, -test_mean, label='Validation Loss')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, -train_mean - train_std, -train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, -test_mean - test_std, -test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Loss.png")
        plt.show()
        # ACCURACY======================

        # Generate the training and validation scores for each epoch
        plt.figure(1, figsize=(4.8, 3.6))
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X, self.y, cv=5)

        # Calculate the mean and standard deviation for the training scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Calculate the mean and standard deviation for the test scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Plot the training and validation scores
        plt.plot(train_sizes, train_mean, label='Training Score')
        plt.plot(train_sizes, test_mean, label='Validation Score')

        # Add the standard deviation bands to the plot
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

        # Add a legend and show the plot
        plt.legend()
        plt.savefig("TempPlots/Accuracy.png")
        plt.show()

        # CM======================
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

        cm = confusion_matrix(self.y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
        disp.plot()
        plt.savefig("TempPlots/CM.png")
        plt.show()
        accuracy = accuracy_score(self.y_test, predictions, normalize=True, sample_weight=None)
        print(accuracy)
        pickle.dump(self.model, open('DTCKF.h5', 'wb'))
        self.GoruntuYukle()

    def EgitimVerisetiniAktar(self, data):
        X_train = data
        # self.X
        numColumnsX = X_train.shape[1]
        NumRowsX = X_train.shape[0]

        self.tblEgitim.setColumnCount(numColumnsX)
        self.tblEgitim.setRowCount(NumRowsX)
        self.tblEgitim.setHorizontalHeaderLabels(self.headers)
        # self.tableWidget.setHorizontalHeaderLabels(self.headers)

        for i in range(NumRowsX):
            for j in range(numColumnsX):
                self.tblEgitim.setItem(i, j, QTableWidgetItem(str(X_train[i, j])))

        self.tblEgitim.resizeColumnsToContents()
        self.tblEgitim.resizeRowsToContents()

    def TestVerisetiniAktar(self, data):
        Y_Train = data
        # self.X_test
        numColumnsY = Y_Train.shape[1]
        NumRowsY = Y_Train.shape[0]

        self.tblTest.setColumnCount(numColumnsY)
        self.tblTest.setRowCount(NumRowsY)
        self.tblTest.setHorizontalHeaderLabels(self.headers)

        for i in range(NumRowsY):
            for j in range(numColumnsY):
                self.tblTest.setItem(i, j, QTableWidgetItem(str(Y_Train[i, j])))

        self.tblTest.resizeColumnsToContents()
        self.tblTest.resizeRowsToContents()

    def Predict(self, model):
        self.KullaniciGirdisiniAktar()
        self.diabetes = np.reshape(self.diabetes, (1, -1))
        self.diabetes = self.sc.transform(self.diabetes)
        singlePrediction = model.predict(self.diabetes)
        print(singlePrediction)
        singlePrediction = (np.rint(singlePrediction)).astype(int)
        print(singlePrediction)

    def GoruntuYukle(self):
        self.imCM.setPixmap(QtGui.QPixmap(self.imageCM))
        self.imLoss.setPixmap(QtGui.QPixmap(self.imageLoss))
        self.imAccuracy.setPixmap(QtGui.QPixmap(self.imageAccuracy))

        # 2 tane X_Train var bir tanesi table widget'a yazarken geliyor onu normal x_train ile değiştirmek lazım.

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
