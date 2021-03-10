#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:59:47 2020

@author: nahuel
"""

from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import numpy as np
import mne
from mne.decoding import CSP
from mne.channels import read_layout
#from loaddata import *
from sklearn import metrics as met
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)


def loadDatos(cnt_file, events_file):
    #Seteamos frecuencia de muestreo Cyton
    freq=250
    #Se carga la matriz de datos
    data_cnt= np.load(cnt_file)
    data_cnt=data_cnt.transpose()
    
    #Se carga los nombre de los caneles
    ch_names_txt = open('DATA/ch_names.txt', "r")
    ch_names = ch_names_txt.read().split(',')
    for i in range(len(ch_names)):
        ch_names[i]=ch_names[i].strip()
    info = mne.create_info(ch_names, freq, 'eeg')
    raw = mne.io.RawArray(data_cnt, info, first_samp=0, copy='auto', verbose=None)
    
    #Se carga la matriz de eventos
    events= np.load(events_file)

    return raw, events

def main():
        
    low_freq, high_freq = 7., 30.
    tmin, tmax = 1., 2.
    
    # event_id
    event_id = {'right': 1, 'foot': 0}
    
    acurracy = []
        
    #Se carga set de datos crudos
    raw, events = loadDatos('DATA/data.npy', 'DATA/events.npy')
    
    #Seleccionamos los canales a utilizar
    raw.pick_channels(['Fp1', 'Fp2', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    #print('raw select: ', raw.shape)
    
    #Seteamos la ubicacion de los canales segun el 
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Se aplica filtros band-pass
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    
    #Se carga eventos   
    #events = creatEventsArray(fp[sujeto])
    #events= np.load('DATA/events.npy')
    
    #Se genera las epocas con los datos crudos y los eventos
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    
    #Se carga target (convierte 1 -> -1 y 2 -> 0 )
    target = epochs.events[:, -1] - 2
    
    #Lo convierte a matriz numpy
    epochs_data = epochs.get_data()
    #print(epochs_data.shape )
    
    #Se crea set de de pruebas y test
    X_train, X_test, y_train, y_test = train_test_split(epochs_data, target, test_size=0.2, random_state=0)
    
    #Clasificadores del modelo
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    
    #Modelo utiliza CSP y LDA
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    
    #Entrenamiento del modelo
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    
    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, target)
    csp.plot_patterns(epochs.info, ch_type='eeg', size=1.5)
    
    #Resultados
    result=clf.predict(X_test)
    
    print(X_train.shape)
    print(X_test.shape)

    
    #Archivo de salida
    fout=open("output.txt","a")
    fout.write(str(met.confusion_matrix(y_test, result)) + "\n")
    fout.write(str( met.classification_report(y_test, result) ))
    fout.write("\n")
    fout.close()
    
if __name__ == "__main__":
    main()