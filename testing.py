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

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *
"""
def loadDatos(data):
    #Seteamos frecuencia de muestreo Cyton
    freq=250
    #Se carga la matriz de datos
    data_cnt=data
    #Se carga los nombre de los caneles
    ch_names_txt = open('ch_names.txt', "r")
    ch_names = ch_names_txt.read().split(',')
    for i in range(len(ch_names)):
        ch_names[i]=ch_names[i].strip()
    info = mne.create_info(ch_names, freq, 'eeg')
    #print(ch_names)
    raw = mne.io.RawArray(data_cnt, info, first_samp=0, copy='auto', verbose=None)
    
    return raw
"""

###
### CONSTANTES DE ARCHIVOS INPUT
###
path_raiz = 'DATA/'
#datos
name = 'T10'
name_realtime = 'R11'
#modelo
name_model = 'T10'
#path datos
path = path_raiz + name
#path modelo
path_model = path_raiz + name_model
#path realtime
path_realtime = path_raiz +name_realtime
###

low_freq, high_freq = 7., 30.
tmin, tmax = 1., 2.

###
### DATOS CRUDOS
###
#total_data= np.load(path + '/total_data.npy')
#print('total_data: ',total_data.shape)

data= np.load(path + '/data.npy')
print('data: ',data.shape)

lista_ts= np.load(path + '/lista_ts.npy')
print('lista_ts: ',lista_ts.shape)

labels= np.load(path + '/labels.npy')
print('labels: ',labels.shape)

events= np.load(path + '/events.npy')
print('events: ', events.shape)
print(type(events))
#print(lista_ts[722])
#print(labels[0][0])


"""
i=0
for pos, x, lab in events:
    print(  datetime.fromtimestamp( labels[i][0]) , ' - ', datetime.fromtimestamp(lista_ts[pos]) , labels[i][1], '-', lab )
    i=i+1
"""


###
### SET DE DATOS TRAIN & TEST
###
#X_train= np.load(path + '/X_train.npy')
#y_train= np.load(path + '/y_train.npy')
X_test = np.load(path + '/X_test.npy')
y_test = np.load(path + '/y_test.npy')

#print("x_train: ", X_train.shape)
#print("y_train: ",y_train.shape)
print("X_test: ", X_test.shape)
print("X_test: ", type( X_test[0][0][0]) )
print("y_test: ",y_test.shape)

#Cargamos modelo
model = joblib.load(path_model + '/model.pkl')

#score = model.score(X_train, y_train)
#print("Score entrenamiento: ", score)

#Resultados
result=model.predict(X_test)
print("y_test: ",y_test)
print("result: ",result)

#Variables report
ts = time.time()
matriz=met.confusion_matrix(y_test, result)
report=met.classification_report(y_test, result)

#Mostrar report
print(ts, ' - ', datetime.fromtimestamp(ts))
print(matriz)
print(report)



data_realtime= np.load(path_realtime + '/data.npy')
data_input = np.load(path_realtime + '/data_input.npy')
#data_input2= np.load(path_realtime + '/data_input.npy')


"""
for i in range(32):
    #print("X: ", X_test[i].shape)
    raw=loadDatos(X_test[i])
    events=np.array( [ [1, 0, y_test[i] ] ])
    raw.plot(scalings='auto', n_channels=8, events=events)
"""
    

#print("realtime: ", data_realtime.shape)
print("realtime 0: ", data_input.shape)

i=1
total_data=None
data=None
for x in data_input:    
    if (i % 3) == 0:
        print(i,': ', x.shape, ' - total')
        data = np.append(data, x, axis=1)
        if total_data is None:
            total_data = np.array([data])
        else:
            total_data = np.append(total_data, np.array([data]), axis=0)
        data=None
    else:
        print(i,': ', x.shape)
        if data is None:
            data = x
        else:
            data = np.append(data, x, axis=1)
    i=i+1


print("data 0: ", data.shape)
print("total_data 0: ", total_data.shape)

data_input=total_data

for i in range(5):
    print("data_input: ", data_input[i].shape)
    
    raw=loadDatos(data_input[i], 'ch_names.txt')
    #Seleccionamos los canales a utilizar
    raw.pick_channels(['P3', 'P4', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    
    #Seteamos la ubicacion de los canales segun el 
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Se aplica filtros band-pass
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    
    #events=np.array( [ [1, 0, y_test[i] ] ])
    raw.plot(scalings='auto', n_channels=8)
    
    data_out = raw.get_data()
    a= data_out[:,:251]
    b= data_out[:,251:502]
    c= data_out[:,502:]
    d = np.array([a, b, c ])
    print(d.shape)
    result=model.predict(d)
    print(result)
    print(result.mean())


#raw=loadDatos(data_realtime)
#raw.plot(scalings='auto', n_channels=8)


#result=model.predict(data_input)
#print(result)