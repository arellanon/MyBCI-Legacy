#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:57:01 2021

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
        print(path)
        #print("board_id: ", board_id)
        #print("sampling_rate: ", self.sampling_rate)
        #print("egg channels: ",self.eeg_channels)
    
    def run (self):
        sleep_time = 1
        count=1
        total_data = None
        posiciones = None
        while self.keep_alive:
            time.sleep(sleep_time)
            data = self.board.get_board_data()
            count=count+1
            if total_data is None:
                total_data = data                
            else:
                total_data = np.append(total_data, data, axis=1)
        #print(count, ': Data Shape ', total_data.shape, ' timestamp: ', datetime.fromtimestamp(total_data[22][0]) )
               
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
        data = total_data[1:9, :]
        
        #Guardamos los datos crudos
        np.save(self.path + '/data.npy', data)
        np.save(self.path + '/events.npy', events)
        np.save(self.path + '/total_data.npy', total_data)
        np.save(self.path + '/lista_ts.npy', lista_ts)
        np.save(self.path + '/labels.npy', labels)
        
        
def test():
    run_n = 1
    trial_per_run = 40
    time_trial = 4
    time_fixation = 3
    time_pause = 4
    time_pause_per_run = 20
    labels=None
    
    #variables para sonido beep
    duration = 1  # seconds
    freq = 440  # Hz pitido
    
    for i in range(run_n):
        print('\nCorrida N#: ', i)
        #Se crea lista de stack
        stack = []
        left  = [0] * (trial_per_run // 2)
        rigth = [1] * (trial_per_run // 2)    
        stack = left + rigth
        print(stack)
        random.shuffle(stack)
        #print(stack)
        #time pause per run
        time.sleep(time_pause_per_run)
        for x in stack:
            #time fixation
            for j in range(time_fixation):
                print('.', end="")
                time.sleep(1)
            #time beep
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) #beep
            ts = time.time()
            print()
            print(x, ' ', ts, ' - ', datetime.fromtimestamp(ts))
            label=np.array( [ [ts], [x] ] )
            if labels is None:
                labels = label
            else:
                labels = np.append(labels, label, axis=1)
            
            for j in range(time_trial):
                if x == 0:
                    print('+', end="")
                else:
                    print('>', end="")
                time.sleep(1)
            time.sleep(time_pause)
    
    labels=labels.transpose() #realizamos la traspuesta ts x event
    return labels


def main ():
    #Calculamos name del directorio nuevo.
    path_raiz='DATA/'
    name = new_name( path_raiz, 'T')
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
        #time.sleep(60)
        labels = test()
    finally:
        data_thead.labels=labels
        data_thead.keep_alive = False
        data_thead.join()
        
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()
