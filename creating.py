#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:39:10 2021

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
from libb import *
import os
from os import listdir
from os.path import isfile, isdir

#sklearn
#from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.pipeline import Pipeline
#from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import joblib


path_raiz = 'DATA/'
#datos
lista = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T13']

total_data=None
total_lista_ts=None
total_labels=None

for name in lista:
    path = path_raiz + name
    data= np.load(path + '/data.npy')
    lista_ts= np.load(path + '/lista_ts.npy')    
    labels= np.load(path + '/labels.npy')
    print(path)
    print('data: ',data.shape)
    print('labels: ',labels.shape)
    print('lista_ts: ',lista_ts.shape)
    
    if total_data is None:
        total_data = data                
    else:
        total_data = np.append(total_data, data, axis=1)
        
    if total_lista_ts is None:
        total_lista_ts = lista_ts                
    else:
        total_lista_ts = np.append(total_lista_ts, lista_ts, axis=0)
        
    if total_labels is None:
        total_labels = labels                
    else:
        total_labels = np.append(total_labels, labels, axis=0)
        

print('total_data: ',total_data.shape)
print('total_lista_ts: ',total_lista_ts.shape)
print('total_labels: ',total_labels.shape)


posiciones = None
i=0
#Buscamos posicion del evento por proximidad ts
for x in total_labels:
    resta = abs(total_lista_ts - x[0])
    pos = np.where(min(resta) == resta)[0]
    print(i,': ', pos)
    if posiciones is None:
        posiciones = pos
    else:
        posiciones = np.append(posiciones, pos)
    i=i+1

print(type(posiciones))
print(posiciones.shape)
#Con las posiciones creamos matriz de eventos pos x zero x event
events = np.zeros((len(total_labels) , 3), int)
events[:, 0] = posiciones.astype(int)
events[:, 2] = total_labels[:,1].astype(int)


name_write=new_name( path_raiz, 'T')
path_write =path_raiz + name_write
#Creamos directorio
print(path_write)
os.makedirs(path_write, exist_ok=True)

#Guardamos los datos crudos
np.save(path_write + '/data.npy', total_data)
np.save(path_write + '/events.npy', events)
np.save(path_write + '/lista_ts.npy', total_lista_ts)
np.save(path_write + '/labels.npy', total_labels)