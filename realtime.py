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
        name_model = 'T6'
        #path modelo
        path_model = path_raiz + name_model
        #Cargamos modelo
        model = joblib.load(path_model + '/model.pkl')
        
        sleep_time = 1
        count=1
        total_data = None
        posiciones = None
        while self.keep_alive:
            time.sleep(sleep_time)            
            data = self.board.get_board_data()
            #timestamp
            ts=data[22][0]
            #seleccionamos channels
            data = data[1:9, :]
            data_input = np.resize(data, (1, 8, 251))
            result=model.predict(data_input)
            #print(count, ': Data Shape ', data_input.shape, ' timestamp: ', datetime.fromtimestamp(ts), ' result: ', result )
            print(data_input)
            print(result)
            count=count+1
            
            """
            if total_data is None:
                total_data = data                
            else:
                total_data = np.append(total_data, data, axis=1)
            #print(count, ': Data Shape ', total_data.shape, ' timestamp: ', datetime.fromtimestamp(total_data[22][0]) )
            """
            

        """               
        #Seleccionamos lista de timestamps
        lista_ts = total_data[22, :]

        #Recuperamos los labels desde el main()
        labels=self.labels
        #Buscamos posicion del evento por proximidad ts
        for x in labels:
            resta = abs(lista_ts - x[0])
            pos = np.where(min(resta) == resta)[0]
            if posiciones is None:
                posiciones = pos
            else:
                posiciones = np.append(posiciones, pos)
                
        #Con las posiciones creamos matriz de eventos pos x zero x event
        events = np.zeros((len(labels) , 3), int)
        events[:, 0] = posiciones.astype(int)
        events[:, 2] = labels[:,1].astype(int)
        
        #Seleccionamos los canales egg        
        data = total_data[1:9, :].transpose()
        """     

def main ():    
    BoardShim.enable_board_logger()

    params = BrainFlowInputParams ()
    params.serial_port = '/dev/ttyUSB0'
    board_id = BoardIds.CYTON_BOARD.value
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    path=""
    
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