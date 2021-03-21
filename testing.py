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

###
### CONSTANTES DE ARCHIVOS INPUT
###
path_raiz = 'DATA/'
#datos
name = 'T2'
#modelo
name2 = 'T2'
#path datos
path = path_raiz + name
#path modelo
path2 = path_raiz + name2
###



###
### DATOS CRUDOS
###
total_data= np.load(path + '/total_data.npy')
print('total_data: ',total_data.shape)

data= np.load(path + '/data.npy')
print('data: ',data.shape)

lista_ts= np.load(path + '/lista_ts.npy')
print('lista_ts: ',lista_ts.shape)

labels= np.load(path + '/labels.npy')
print('labels: ',labels.shape)

events= np.load(path + '/events.npy')
print('events: ', events.shape)
#print(events)
#print(lista_ts[722])
#print(labels[0][0])

i=0
for pos, x, lab in events:
    print(  datetime.fromtimestamp( labels[i][0]) , ' - ', datetime.fromtimestamp(lista_ts[pos]) , labels[i][1], '-', lab )
    i=i+1



###
### SET DE DATOS TRAIN & TEST
###
X_train= np.load(path + '/X_train.npy')
y_train= np.load(path + '/y_train.npy')
X_test = np.load(path + '/X_test.npy')
y_test = np.load(path + '/y_test.npy')

print("x_train: ", X_train.shape)
print("y_train: ",y_train.shape)
print("X_test: ", X_test.shape)
print("y_test: ",y_test.shape)

#Cargamos modelo
model = joblib.load(path2 + '/model.pkl')

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