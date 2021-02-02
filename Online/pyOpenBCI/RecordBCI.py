#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 01:21:50 2021

@author: nahuel
"""
import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def main ():
    params = BrainFlowInputParams()
    params.serial_port = '/dev/ttyUSB0'
    
    board_id = BoardIds.CYTON_BOARD.value

    board = BoardShim(board_id, params)    
    board.prepare_session()    
    #get_eeg_channels (board_id)

    # board.start_stream () # use this for default options
    board.start_stream(45000)
    time.sleep(1)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data() # get all data and remove it from internal buffer
    eeg_channels = board.get_eeg_channels(board_id)
    emg_channels = board.get_emg_channels(board_id)
    sampling_rate = board.get_sampling_rate(board_id)
    timestamp_channel = board.get_timestamp_channel(board_id)

    board.stop_stream()
    board.release_session()

    print(data)
    print(eeg_channels)
    print(emg_channels)
    print(sampling_rate)
    print(data[timestamp_channel][0])
    print(data[timestamp_channel][100])
    
    #np.save('test1.npy', data)
    #print(data.shape)
    #print(data[0])
    

if __name__ == "__main__":
    main()