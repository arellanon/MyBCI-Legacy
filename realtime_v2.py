#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 00:59:06 2021

@author: nahuel
"""
import time
import brainflow
import numpy as np
import threading
import random
import os
from datetime import datetime
from os import listdir
from os.path import isfile, isdir
from libb import *
import joblib

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

class DataThread (threading.Thread):
    
    def __init__ (self, board, board_id, path):
        threading.Thread.__init__ (self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.labels = None
        self.board = board
        self.path = path
    
    def run (self):
        low_freq, high_freq = 7., 30.
        path_raiz = 'DATA/'
        name_model = 'T10'
        #path modelo
        path_model = path_raiz + name_model
        #Cargamos modelo
        model = joblib.load(path_model + '/model.pkl')
        
        sleep_time = 1
        count=1
        total_data = None
        while self.keep_alive:
            time.sleep(sleep_time)
            #data = self.board.get_board_data()
            #me guardo 3 seg.
            data = self.board.get_current_board_data(251)
            c = self.board.get_board_data_count()
            print("dim: ", data.shape, "cant: ", c)
            """
            if len(data[0]) == 753:
                #timestamp
                ts=data[22][0]
                #Seleccionamos los canales egg
                data = data[1:9, :]
                
                raw=loadDatos(data, 'ch_names.txt')
                #Seleccionamos los canales a utilizar
                raw.pick_channels(['P3', 'P4', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
                
                #Seteamos la ubicacion de los canales segun el 
                montage = make_standard_montage('standard_1020')
                raw.set_montage(montage)
                
                # Se aplica filtros band-pass
                raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
                data = raw.get_data()
                
                #divido los datos en 3 set, 1 por seg.
                data1= data[:,:251]
                data2= data[:,251:502]
                data3= data[:,502:]
                
                data = np.array([data1, data2, data3])
                #data = np.array([data_out])
                result=model.predict(data)
                
                #print(count, ': ', data.shape, ' - ', datetime.fromtimestamp(ts) )
                print(count, ': ', data.shape, ' - ', datetime.fromtimestamp(ts), ' - ', result )
                count=count+1
                if total_data is None:
                    total_data = data                
                else:
                    total_data = np.append(total_data, data, axis=0)
            """
        #print(count, ': Data Shape ', total_data.shape, ' timestamp: ', datetime.fromtimestamp(ts) )
        print(self.path)
        np.save(self.path + '/data.npy', data)
        np.save(self.path + '/data_input.npy', total_data)

def main ():
    #Calculamos name del directorio nuevo.
    path_raiz='DATA/'
    name = new_name( path_raiz, 'R')
    path = path_raiz + name
    #Creamos directorio
    os.makedirs(path, exist_ok=True)
    
    BoardShim.enable_board_logger()

    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'
    board_id = BoardIds.CYTON_BOARD.value
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    data_thead = DataThread(board, board_id, path)
    data_thead.start()
    try:
        time.sleep(30)
    finally:
        data_thead.keep_alive = False
        data_thead.join()
        
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()