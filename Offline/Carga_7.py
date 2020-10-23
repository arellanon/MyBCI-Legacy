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

import mne
from mne.decoding import CSP
from mne.channels import read_layout
from loaddata import *
from sklearn import metrics as met
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

fp = {
      'aa' : {
          'cnt': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/data_set_IVa_aa_cnt.txt',
          'mrk': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/data_set_IVa_aa_mrk.txt',
          'lab': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aa/TRUE_LABELS.txt',
          'chn': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/ch_names.txt',
          'pos':168,
          'freq': 100
          },
       'al' : {
          'cnt': '../../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/data_set_IVa_al_cnt.txt',
          'mrk': '../../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/data_set_IVa_al_mrk.txt',
          'lab': '../../DATA/Competencia BCI/III/Dataset_IV/txt/al/TRUE_LABELS.txt',
          'chn': '../../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/ch_names.txt',
          'pos':224,
          'freq': 100
       },
       'av' : {
          'cnt': '../../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/data_set_IVa_av_cnt.txt',
          'mrk': '../../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/data_set_IVa_av_mrk.txt',
          'lab': '../../DATA/Competencia BCI/III/Dataset_IV/txt/av/TRUE_LABELS.txt',
          'chn': '../../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/ch_names.txt',
          'pos':84,
          'freq': 100
       },
       'aw' : {
          'cnt': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/data_set_IVa_aw_cnt.txt',
          'mrk': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/data_set_IVa_aw_mrk.txt',
          'lab': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aw/TRUE_LABELS.txt',
          'chn': '../../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/ch_names.txt',
          'pos':56,
          'freq': 100
          },
       'ay' : {
          'cnt': '../../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/data_set_IVa_ay_cnt.txt',
          'mrk': '../../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/data_set_IVa_ay_mrk.txt',
          'lab': '../../DATA/Competencia BCI/III/Dataset_IV/txt/ay/TRUE_LABELS.txt',
          'chn': '../../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/ch_names.txt',
          'pos':28,
          'freq': 100
       }
}

#fp = { 'aa': fp['aa'] }

low_freq, high_freq = 7., 30.
tmin, tmax = 1., 2.

# event_id
event_id = {'right': 1, 'foot': 2}

acurracy = []
fout=open("output.txt","w")
fout.close()

#sujeto='ay'

for sujeto in fp:
    #Se carga set de datos crudos
    raw = creatRawArray(fp[sujeto])
    
    #print(raw['ch_names'])
    #Seleccionamos los canales a utilizar
    raw.pick_channels(['Fp1', 'Fp2', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    
    #Seteamos la ubicacion de los canales segun el 
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Se aplica filtros band-pass
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    
    #Se carga eventos
    events = creatEventsArray(fp[sujeto])
    
    #Se genera las epocas con los datos crudos y los eventos
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    
    #Se carga target (convierte 1 -> -1 y 2 -> 0 )
    target = epochs.events[:, -1] - 2
    
    #Lo convierte a matriz numpy
    epochs_data = epochs.get_data()
    
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
        
    fout=open("output.txt","a")
    fout.write("Sujeto: " + sujeto + "\n")
    fout.write(str(met.confusion_matrix(y_test, result)) + "\n")
    fout.write(str( met.classification_report(y_test, result) ))
    fout.write("\n")
    fout.close()
    
    acurracy.append("Acurracy - Sujeto " + sujeto + ": "+ str(met.accuracy_score(y_test, result) ) )
    
    print(met.classification_report(y_test, result))

print(acurracy)
fout=open("output.txt","a")
for x in acurracy:
    fout.write(x + "\n")
fout.write("\n")
fout.close()