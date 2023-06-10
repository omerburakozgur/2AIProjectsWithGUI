# Import Libraries
import shutil

import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import keras
from tqdm import tqdm
import cv2
import sklearn
import skimage
from skimage.transform import resize
import random
from skimage.color import rgb2gray
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import sys
import joblib
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from sklearn.model_selection import train_test_split, KFold, learning_curve
from CNNArayuz import Ui_MainWindow
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from keras.preprocessing import image
from keras.models import Model

sns.set()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # %% Değişkenler
        self.setupUi(self)
        self.train = "/data/forest_fire/fire"
        self.test = "/data/forest_fire/nofire"
        self.folder = "data/forest_fire"
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.gotData = False
        self.predictions = None
        self.imageLoaded = False
        self.imageCM = "CNNTempPlots/CM.png"
        self.imageLoss = "CNNTempPlots/Loss.png"
        self.imageAccuracy = "CNNTempPlots/Accuracy.png"
        self.pixmap = None
        self.userFile = None
        #QMainWindow.showMaximized(self)
        self.groupBox_5.setAutoFillBackground(True)
        self.y_test_pred = None
        self.History = None
        self.y_pred = None
        self.x = []
        self.y = []
        self.batch_size = 16
        self.epochs = 3
        print(tf.__version__)
        self.btnModelEgit.clicked.connect(self.ModelEgit)
        self.btnModelKullan.clicked.connect(self.ModelTahmin)
        self.btnGoruntuSec.clicked.connect(self.GoruntuSec)

        # Yapılacaklar
        # Data Augmentation
        # Validation otomatik ayıracağız
        # Val_Loss grafiğini göstermek lazım

    def get_data(self):
        for folderName in os.listdir(self.folder):
            if not folderName.startswith("."):
                if folderName in ["nofire"]:
                    label = 0
                elif folderName in ["fire"]:
                    label = 1
                else:
                    label = 2
                for image_filename in tqdm(os.listdir(self.folder + "/" + folderName + "/")):
                    img_file = cv2.imread(self.folder + "/" + folderName + "/" + image_filename)
                    if img_file is not None:
                        img_file = skimage.transform.resize(img_file, (227, 227, 3), mode="constant", anti_aliasing=True)
                        img_arr = np.asarray(img_file, float)
                        self.x.append(img_arr)
                        self.y.append(label)
        x = np.array(self.x)
        y = np.array(self.y)
        self.gotData = True
        return x, y

    def TestTrainSplit(self):
        #Test Train Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=True)

    def Model1TTS(self):
        self.TestTrainSplit()
        #Model
        self.model = keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPool2D(2, 2))

        self.model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(2, 2))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.MaxPool2D(2, 2))

        self.model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self.model.add(tf.keras.layers.MaxPool2D(2, 2))
        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(512, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        self.history = self.model.fit(self.X_train,self.y_train,validation_data=(self.X_test,self.y_test),batch_size=self.batch_size,
                            epochs=self.epochs,verbose=1,callbacks=[early_stopping])

        score = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = self.model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        self.model.save("Model1HLD.h5")
        return self.model

    def Model1KF(self):
        # Model
        model = keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(2, 2))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(2, 2))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(2, 2))
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Iterate over the folds
        self.x = np.array(self.x)
        self.y = np.array(self.y)

        #X = self.x.reshape(1900, -1)
        #y = self.y.reshape(1900, 1)
        #X = self.x
        #y = self.y


        # Set the number of folds for cross-validation
        k = 4
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        # Loop through the folds
        for train_index, eval_index in kfold.split(self.x):
            # Split the data into training and validation sets
            self.X_train, self.X_test = self.x[train_index], self.x[eval_index]
            self.y_train, self.y_test = self.y[train_index], self.y[eval_index]


            self.history = model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test), epochs=self.epochs, batch_size=self.batch_size)
            model.save("Model1KF.h5")



        score = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        model.save("Model1KF.h5")
        return model

    def ResNetTTS(self):
        self.TestTrainSplit()
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

        # Freeze the weights of the base model
        base_model.trainable = False

        # Add a new fully connected layer on top of the base model to perform the classification task
        inputs = keras.Input(shape=(227, 227, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        # Compile the model with a binary cross-entropy loss and an Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=5,
            restore_best_weights=True)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        self.history = model.fit(self.X_train,self.y_train,validation_data=(self.X_test,self.y_test),batch_size=self.batch_size,
                            epochs=self.epochs,verbose=1,callbacks=[early_stopping])

        score = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        model.save("ResNetHLD.h5")
        return model

    def ResNetKF(self):
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

        # Freeze the weights of the base model
        base_model.trainable = False
        inputs = keras.Input(shape=(227, 227, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('bestmodel.hdf5', monitor='loss', save_best_only=True)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        #tscv = TimeSeriesSplit(n_splits=5)
        k = 4
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        # Loop through the folds
        for train_index, eval_index in kfold.split(self.x):
            # Split the data into training and validation sets
            self.X_train, self.X_test = self.x[train_index], self.x[eval_index]
            self.y_train, self.y_test = self.y[train_index], self.y[eval_index]

            self.history = model.fit(self.X_train, self.y_train,validation_data=(self.X_test,self.y_test), epochs=self.epochs, batch_size=self.batch_size)
            model.save("ResNetKF.h5")
            #, callbacks=[modelCheckpoint]


        score = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        model.save("ResNetKF.h5")
        return model

    def VGG19TTS(self):
        self.TestTrainSplit()
        base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)

        # Freeze the weights of the base model
        base_model.trainable = False

        # Add a new fully connected layer on top of the base model to perform the classification task
        inputs = keras.Input(shape=(227, 227, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        # Compile the model with a binary cross-entropy loss and an Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="accuracy",
            patience=5,
            restore_best_weights=True)

        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        self.history = model.fit(self.X_train,self.y_train,validation_data=(self.X_test,self.y_test),batch_size=self.batch_size,
                            epochs=self.epochs,verbose=1,callbacks=[early_stopping])

        score = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)
        print(score)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        model.save("VGG19HLD.h5")
        return model

    def VGG19KF(self):
        base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False)

        # Freeze the weights of the base model
        base_model.trainable = False
        inputs = keras.Input(shape=(227, 227, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #modelCheckpoint = tf.keras.callbacks.ModelCheckpoint('bestmodel.hdf5', monitor='loss', save_best_only=True)

        self.x = np.array(self.x)
        self.y = np.array(self.y)

        #tscv = TimeSeriesSplit(n_splits=5)
        k = 4
        kfold = KFold(n_splits=k, shuffle=True, random_state=1)
        # Loop through the folds
        for train_index, eval_index in kfold.split(self.x):
            # Split the data into training and validation sets
            self.X_train, self.X_test = self.x[train_index], self.x[eval_index]
            self.y_train, self.y_test = self.y[train_index], self.y[eval_index]

            self.history = model.fit(self.X_train, self.y_train,validation_data=(self.X_test,self.y_test), epochs=self.epochs, batch_size=self.batch_size)
            model.save("VGG19KF.h5")
            #, callbacks=[modelCheckpoint]


        score = model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size, verbose=1)

        self.y_test_pred = model.predict(self.X_test)
        self.y_test_pred = np.round(self.y_test_pred)
        self.y_pred = (self.y_test_pred > 0.5)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print(self.history.history.keys())
        self.GrafikCiz()
        model.save("VGG19KF.h5")
        return model

    def GrafikCiz(self):
        plt.figure(figsize=(5, 3.6))
        plt.plot(self.history.history['accuracy'], color='r')
        plt.plot(self.history.history['val_accuracy'], color='b')
        plt.title('Model Accuracy', weight='bold', fontsize=16)
        plt.ylabel('Accuracy', weight='bold', fontsize=14)
        plt.xlabel('Epoch', weight='bold', fontsize=14)
        plt.ylim(0.2, 1.0)
        plt.xticks(weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)
        plt.legend(['train', 'val'], loc='lower left', prop={'size': 14})
        plt.grid(color='y', linewidth='0.5')
        plt.plot()
        plt.savefig("CNNTempPlots/Accuracy.png")

        plt.figure(figsize=(5, 3.6))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.title("Epoch - Loss Graph", weight='bold', fontsize=16)
        plt.xlabel("Epochs", weight='bold', fontsize=14)
        plt.ylabel("Loss", weight='bold', fontsize=14)
        plt.legend(['Loss'], loc='upper right', prop={'size': 14})
        plt.plot()
        plt.savefig("CNNTempPlots/Loss.png")

        names = ['No Fire', 'Fire']
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        f, ax = plt.subplots(figsize=(5, 3.6))
        sns.heatmap(cm, annot=True, linewidth=.5, linecolor="r", fmt=".0f", ax=ax)
        plt.title("CNN", size=25)
        plt.xlabel("Predictions")
        plt.ylabel("Actuals")
        ax.set_xticklabels(names)
        ax.set_yticklabels(names)
        plt.plot()
        plt.savefig("CNNTempPlots/CM.png")
        plt.show()

        TN, TP, FP, FN = cm[0, 0], cm[1, 1], cm[0, 1], cm[1, 0]
        precision = TP / (TP + FP)
        sensitivity = TP / (TP + FN)
        F1 = (2 * precision * sensitivity) / (precision + sensitivity)
        accuracy = (TP + TN) / (TP + FP + TN + FN) * 100
        specificity = TN / (TN + FP)

        accuracy = round(accuracy, 2)
        precision = round(precision, 2)
        F1 = round(F1, 2)
        sensitivity = round(sensitivity, 2)
        specificity = round(specificity, 2)

        print("Accuracy: ", accuracy)
        print("Sensitivity: ", sensitivity)
        print("Precision: ", precision)
        print("Specificity: ", specificity)
        print("F1: ", F1)

        self.lblAccuracy.setText(f"Accuracy: {accuracy}")
        self.lblSensitivity.setText(f"Sensitivity: {sensitivity}")
        self.lblPrecision.setText(f"Precision: {precision}")
        self.lblSpecificity.setText(f"Specificity: {specificity}")
        self.lblF1.setText(f"F1: {F1}")

        self.imCM.setPixmap(QtGui.QPixmap(self.imageCM))
        self.imLoss.setPixmap(QtGui.QPixmap(self.imageLoss))
        self.imAccuracy.setPixmap(QtGui.QPixmap(self.imageAccuracy))

    def ModelEgit(self):
        self.epochs = int(self.txtEpoch.text())
        self.batch_size = int(self.txtBatch.text())
        if self.gotData == False:
            self.get_data()
        if self.imageLoaded == True:
            self.lblError.setText("Eğitim İşlemi Devam Ediyor, Lütfen Bekleyin...")
            if self.cmbModel.currentIndex() == 0:
                model = self.Model1TTS()
                self.Predict(model)
            elif self.cmbModel.currentIndex() == 1:
                model = self.Model1KF()
                self.Predict(model)
            elif self.cmbModel.currentIndex() == 2:
                model = self.ResNetTTS()
                self.Predict(model)
            elif self.cmbModel.currentIndex() == 3:
                model = self.ResNetKF()
                self.Predict(model)
            elif self.cmbModel.currentIndex() == 4:
                model = self.VGG19TTS()
                self.Predict(model)
            elif self.cmbModel.currentIndex() == 5:
                model = self.VGG19KF()
                self.Predict(model)
        else:
            self.lblError.setText("Lütfen Görüntü Seçiniz!")

    def ModelTahmin(self):
        if self.imageLoaded == True:
            if self.cmbModel.currentIndex() == 0:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\Model1HLD\Model1HLD.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1HLD\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1HLD\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1HLD\Accuracy.png"))
            elif self.cmbModel.currentIndex() == 1:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\Model1KF\Model1KF.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1KF\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1KF\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\Model1KF\Accuracy.png"))
            elif self.cmbModel.currentIndex() == 2:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\ResnetHoldout\ResNetHLD.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetHoldout\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetHoldout\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetHoldout\Accuracy.png"))
            elif self.cmbModel.currentIndex() == 3:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\ResnetKFold\ResNetKF.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetKFold\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetKFold\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\ResnetKFold\Accuracy.png"))
            elif self.cmbModel.currentIndex() == 4:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\VGG19Holdout\VGG19HLD.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19Holdout\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19Holdout\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19Holdout\Accuracy.png"))
            elif self.cmbModel.currentIndex() == 5:
                model = tf.keras.models.load_model("D:\AIProjects\CNNModels\VGG19KF\VGG19KF.h5")
                self.Predict(model)
                self.imCM.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19KF\CM.png"))
                self.imLoss.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19KF\Loss.png"))
                self.imAccuracy.setPixmap(QtGui.QPixmap("D:\AIProjects\CNNModels\VGG19KF\Accuracy.png"))
        else:
            self.lblError.setText("Lütfen Görüntü Seçiniz!")

    def GoruntuSec(self):
        fname = QFileDialog.getOpenFileName(self, "Open File", "AIProjects//data", "All Files (*)")
        if len(fname[0])>=2:
            self.pixmap = QPixmap(fname[0])
            print(fname[0])
            self.userFile = fname[0]
            self.lblGoruntu.setPixmap(self.pixmap)
            self.imageLoaded = True
            self.lblError.setText("Görüntü Yükleme Başarılı!")

        else:
            print("Dialog Exit")

    def Predict(self, model):
        img_file = cv2.imread(self.userFile)
        img_file = skimage.transform.resize(img_file, (227, 227, 3), mode="constant", anti_aliasing=True)
        img_arr = np.asarray(img_file, float)
        onr = []
        onr.append(img_arr)
        onr = np.array(onr)
        self.predictions = model.predict(onr)
        print(self.predictions)
        self.predictions = np.rint(self.predictions).astype(int)
        print(self.predictions)
        if self.predictions == 1:
            self.label_5.setText("Sonuç: Orman Yangını Var")
        elif self.predictions == 0:
            self.label_5.setText("Sonuç: Orman Yangını Yok")
        self.lblError.setText("Eğitim/Tahmin İşlemi Başarılı!")




app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

# Yapılacaklar
# Data Augmentation
# Model Checkpoint