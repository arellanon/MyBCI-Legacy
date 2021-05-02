#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
#from loaddata import *

#sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import joblib

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)


def loadDatos(cnt_file, events_file):
    #Seteamos frecuencia de muestreo Cyton
    freq=250
    #Se carga la matriz de datos
    data_cnt= np.load(cnt_file)
    data_cnt=data_cnt.transpose()
    print(data_cnt.shape )
    print('min: ', np.amin( data_cnt ) )
    print('max: ', np.amax( data_cnt ) )
    print('media: ', np.mean( data_cnt ) )
    print('tipo: ', type(data_cnt[0][0]) )
    
    #Se carga los nombre de los caneles
    ch_names_txt = open('ch_names.txt', "r")
    ch_names = ch_names_txt.read().split(',')
    for i in range(len(ch_names)):
        ch_names[i]=ch_names[i].strip()
    info = mne.create_info(ch_names, freq, 'eeg')
    raw = mne.io.RawArray(data_cnt, info, first_samp=0, copy='auto', verbose=None)
    
    #Se carga la matriz de eventos
    events= np.load(events_file)
    print(data_cnt.shape)
    print(events.shape)
    print(events)

    return raw, events

def main():
    path_raiz = 'DATA/'
    name = 'T5'
    path = path_raiz + name
    
    low_freq, high_freq = 7., 30.
    tmin, tmax = 1., 2.
    
    # event_id
    event_id = {'right': 1, 'foot': 0}
    
    acurracy = []
        
    #Se carga set de datos crudos
    raw, events = loadDatos(path + '/data.npy', path +'/events.npy')
    
    #Seleccionamos los canales a utilizar
    raw.pick_channels(['P3', 'P4', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    #print('raw select: ', raw.shape)
    
    #Seteamos la ubicacion de los canales segun el 
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    #raw.plot(scalings='auto', n_channels=8, duration=20)
    raw.plot(scalings='auto', n_channels=8, events=events)
    # Se aplica filtros band-pass
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    
    #raw.plot(block=True)
    #raw.plot(scalings='auto', n_channels=1, duration=20)
    #raw.plot_psd(average=True)
    #raw.plot_sensors(ch_type='eeg')
    
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
    #X_train, X_test, y_train, y_test = train_test_split(epochs_data, target, test_size=0.2, random_state=0)
    
    
if __name__ == "__main__":
    main()