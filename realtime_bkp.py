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
            data = self.board.get_board_data()
            #timestamp
            ts=data[22][0]
            #seleccionamos channels
            data = data[1:9, :]
            #inicializamos data_input
            #data_input = np.resize(data, (1, 8, 251))
            data_input= np.zeros((1, 8, 251))
            X = range(8)
            Y = range(251)
            for x in X:
                for y in Y:      
                    if y < len(data[x]):
                        data_input[0][x][y]=data[x][y]
            #ejecutamos modelo
            result=model.predict(data_input)
            print(count, ': Data Shape ', data_input.shape, ' timestamp: ', datetime.fromtimestamp(ts), ' result: ', result )
            #print(data_input)
            #print(result)
            count=count+1
            
            if total_data is None:
                total_data = data_input                
            else:
                total_data = np.append(total_data, data_input, axis=0)
            ##print(count, ': Data Shape ', total_data.shape, ' timestamp: ', datetime.fromtimestamp(total_data[22][0]) )
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

    params = BrainFlowInputParams ()
    params.serial_port = '/dev/ttyUSB0'
    board_id = BoardIds.CYTON_BOARD.value
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    data_thead = DataThread(board, board_id, path)
    data_thead.start()
    try:
        time.sleep(60)
    finally:
        data_thead.keep_alive = False
        data_thead.join()
        
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()