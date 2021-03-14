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


from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


class DataThread (threading.Thread):
    
    def __init__ (self, board, board_id):
        threading.Thread.__init__ (self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.labels = None
        self.board = board
        print("board_id: ", board_id)
        print("sampling_rate: ", self.sampling_rate)
        print("egg channels: ",self.eeg_channels)
    
    def run (self):
        window_size = 5
        sleep_time = 1
        #points_per_update = window_size * self.sampling_rate
        #print("points_per_update: ", points_per_update)
        count=1
        totalData = None
        posiciones = None
        while self.keep_alive:
            time.sleep(sleep_time)
            #data = self.board.get_current_board_data (int (points_per_update))
            data = self.board.get_board_data()
            #print(count, ":", data.shape)
            #print(count, ': Data Shape ', data.shape, ' timestamp: ', datetime.fromtimestamp(data[22][0]) )
            count=count+1
            if totalData is None:
                totalData = data                
            else:
                totalData = np.append(totalData, data, axis=1)
                
        #print(self.labels)
        #print(count, ': Data Shape ', totalData.shape, ' timestamp: ', datetime.fromtimestamp(totalData[22][0]) )
        
        #guardamos los datos crudos
        np.save('totalData.npy', totalData)
        #seleccionamos los canales egg        
        data=totalData[1:9, :]
        data = data.transpose()
        np.save('data.npy', data)
        #guardamos timestamp
        index_ts=totalData[22, :]
        np.save('index_ts.npy', index_ts)
        
        #recuperamos los labels desde el main()
        labels=self.labels
        #buscamos posicion del evento por proximidad ts
        for x in labels:
            resta = abs(index_ts - x[0])
            pos = np.where(min(resta) == resta)[0]
            if posiciones is None:
                posiciones = pos
            else:
                posiciones = np.append(posiciones, pos)
                
        #Con las posiciones creamos matriz de eventos pos x zero x event
        events = np.zeros((len(labels) , 3), int)
        events[:, 0] = posiciones.astype(int)
        events[:, 2] = labels[:,1].astype(int)
        np.save('events.npy', events)
        
        
def test():
    run_n = 1
    trial_per_run = 4
    time_trial = 7
    time_pause = 0
    labels=None
    
    #variables para sonido beep
    duration = 1  # seconds
    freq = 440  # Hz pitido
    
    stack = []
    left  = [0] * (trial_per_run // 2)
    rigth = [1] * (trial_per_run // 2)    
    stack = left + rigth
    print(stack)
    random.shuffle(stack)
    print(stack)
    
    for i in range(run_n):
        print('Corrida N#: ', run_n)
        for x in stack:
            time.sleep(time_pause)
            ts = time.time()
            print()
            print(x, ' ', ts, ' - ', datetime.fromtimestamp(ts))
            label=np.array( [ [ts], [x] ] )
            if labels is None:
                labels = label
            else:
                labels = np.append(labels, label, axis=1)
                
            #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq)) #beep
            for j in range(time_trial):
                if x == 0:
                    print('<', end="")
                else:
                    print('>', end="")
                time.sleep(1)
    
    labels=labels.transpose() #realizamos la traspuesta ts x event
    np.save('labels.npy', labels)
    return labels

def main ():
    BoardShim.enable_board_logger()
    
    params = BrainFlowInputParams ()
    params.serial_port = '/dev/ttyUSB0'
    board_id = BoardIds.CYTON_BOARD.value
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    data_thead = DataThread(board, board_id)
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