#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:35:22 2021

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
#from loaddata import *

#sklearn
#from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import joblib


path_raiz = 'DATA/'
name = 'T5'
path = path_raiz + name

#X_train= np.load('X_train.npy')
#y_train= np.load('y_train.npy')
X_test= np.load(path + '/X_test.npy')
y_test= np.load(path + '/y_test.npy')

#print(X_train.shape)
#print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#Cargamos modelo
model = joblib.load(path + '/model.pkl')

#score = model.score(X_train, y_train)
#print("Score entrenamiento: ", score)

#Resultados
result=model.predict(X_test)

#Variables report
ts = time.time()
matriz=met.confusion_matrix(y_test, result)
report=met.classification_report(y_test, result)

#Mostrar report
print(ts, ' - ', datetime.fromtimestamp(ts))
print(matriz)
print(report)