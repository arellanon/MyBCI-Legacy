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
from datetime import datetime

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


class DataThread (threading.Thread):
    
    def __init__ (self, board, board_id):
        threading.Thread.__init__ (self)
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(board_id)
        self.keep_alive = True
        self.board = board
        print("board_id: ", board_id)
        print("sampling_rate: ", self.sampling_rate)
        print("egg channels: ",self.eeg_channels)
    
    def run (self):
        window_size = 5
        sleep_time = 1
        points_per_update = window_size * self.sampling_rate
        print("points_per_update: ", points_per_update)
        count=1
        totalData = None
        while self.keep_alive:
            time.sleep(sleep_time)
            data = self.board.get_current_board_data (int (points_per_update))
            #print(data)
            print(count, ': Data Shape ', data.shape, ' timestamp: ', datetime.fromtimestamp(data[22][0]) )
            count=count+1
            if totalData is None:
                totalData = data                
            else:
                totalData = np.append(totalData, data, axis=1)
        print(count, ': Data Shape ', totalData.shape, ' timestamp: ', datetime.fromtimestamp(totalData[22][0]) )
        np.save('data1.npy', totalData)
                
    
def main ():
    BoardShim.enable_board_logger()
    
    #use synthetic board for demo
    params = BrainFlowInputParams ()
    params.serial_port = '/dev/ttyUSB0'
    board_id = BoardIds.CYTON_BOARD.value
    
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    
    data_thead = DataThread(board, board_id)
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