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
          'cnt': '../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/data_set_IVa_aa_cnt.txt',
          'mrk': '../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/data_set_IVa_aa_mrk.txt',
          'lab': '../DATA/Competencia BCI/III/Dataset_IV/txt/aa/TRUE_LABELS.txt',
          'chn': '../DATA/Competencia BCI/III/Dataset_IV/txt/aa/100Hz/ch_names.txt',
          'pos':168,
          'freq': 100
          },
       'al' : {
          'cnt': '../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/data_set_IVa_al_cnt.txt',
          'mrk': '../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/data_set_IVa_al_mrk.txt',
          'lab': '../DATA/Competencia BCI/III/Dataset_IV/txt/al/TRUE_LABELS.txt',
          'chn': '../DATA/Competencia BCI/III/Dataset_IV/txt/al/100Hz/ch_names.txt',
          'pos':224,
          'freq': 100
       },
       'av' : {
          'cnt': '../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/data_set_IVa_av_cnt.txt',
          'mrk': '../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/data_set_IVa_av_mrk.txt',
          'lab': '../DATA/Competencia BCI/III/Dataset_IV/txt/av/TRUE_LABELS.txt',
          'chn': '../DATA/Competencia BCI/III/Dataset_IV/txt/av/100Hz/ch_names.txt',
          'pos':84,
          'freq': 100
       },
       'aw' : {
          'cnt': '../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/data_set_IVa_aw_cnt.txt',
          'mrk': '../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/data_set_IVa_aw_mrk.txt',
          'lab': '../DATA/Competencia BCI/III/Dataset_IV/txt/aw/TRUE_LABELS.txt',
          'chn': '../DATA/Competencia BCI/III/Dataset_IV/txt/aw/100Hz/ch_names.txt',
          'pos':56,
          'freq': 100
          },
       'ay' : {
          'cnt': '../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/data_set_IVa_ay_cnt.txt',
          'mrk': '../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/data_set_IVa_ay_mrk.txt',
          'lab': '../DATA/Competencia BCI/III/Dataset_IV/txt/ay/TRUE_LABELS.txt',
          'chn': '../DATA/Competencia BCI/III/Dataset_IV/txt/ay/100Hz/ch_names.txt',
          'pos':28,
          'freq': 100
       }
}

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
    """
    #Seleccionamos los canales a utilizar
    raw.pick_channels(['Fp1', 'Fp2', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    
    #Seteamos la ubicacion de los canales segun el 
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    """
    
    #raw.plot(scalings='auto', n_channels=8, duration=50)
    
    """    
    mapping={'Fp1': 'eog', 'Fp2': 'eog'}
    raw.set_channel_types(mapping)
    
    eog_projs, _ = compute_proj_eog(raw, n_grad=1, n_mag=1, n_eeg=1, reject=None, no_proj=True)
    """    
    
    # Se aplica filtros band-pass
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    
    # set up and fit the ICA
    """
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(raw)
    ica.exclude = [1, 2]  # details on how we picked these are omitted here
    ica.plot_properties(raw, picks=ica.exclude)
    
    raw.plot(scalings='auto', n_channels=8, duration=50)
    raw.add_proj(eog_projs)
    """
    #raw.plot(scalings='auto', n_channels=8, duration=50)    
    
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
    csp = CSP(n_components=5, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    
    #Modelo utiliza CSP y LDA
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    
    #Entrenamiento del modelo
    clf.fit(X_train, y_train)
    score = clf.score(X_train, y_train)
    
    #Resultados
    result=clf.predict(X_test)
        
    fout=open("output.txt","a")
    fout.write("Sujeto: " + sujeto + "\n")
    fout.write(str(met.confusion_matrix(y_test, result)) + "\n")
    fout.write(str( met.classification_report(y_test, result) ))
    fout.write("\n")
    fout.close()
    
    acurracy.append("Acurracy - Sujeto " + sujeto + ": "+ str(met.accuracy_score(y_test, result) ) )
#    tn, fp, fn, tp = met.confusion_matrix(y_test, result).ravel()
#    print(tn, fp, fn, tp)
    
    print(met.classification_report(y_test, result))

print(acurracy)
fout=open("output.txt","a")
for x in acurracy:
    fout.write(x + "\n")
fout.write("\n")
fout.close()