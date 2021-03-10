#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:35:31 2021

@author: nahuel
"""
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


#a=np.load('test1.npy')
data=np.load('data2.npy')
print(data.shape)

# 23 channels
# 0: indice ( 0 .. 255)
# 1: channel 1 egg
# 2: channel 2 egg
# 3: channel 3 egg
# 4: channel 4 egg
# 5: channel 5 egg
# 6: channel 6 egg
# 7: channel 7 egg
# 8: channel 8 egg
# 9: channel 8 egg
# 10: none
# 11: none
# 12: none
# 13: none
# 14: none
# 15: none
# 16: none
# 17: none
# 18: none
# 19: none
# 20: none
# 21: none
# 22: Timestamp



"""
print(data[1])
print(data[2])
print(data[3])
print(data[4])
print(data[5])
print(data[6])
print(data[7])
print(data[8])

print(data[9])
print(data[10])
"""

# Seleccionamos los 8 channels egg
A=data[22, :]
print(A.shape)


#print(data[0][1024])
np.savetxt('matriz_a.dat', A, fmt='%.4e')



#data = data.transpose()
#print(data.shape)
#print(data[0].shape)


"""
board_id = BoardIds.CYTON_BOARD.value
print(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
print(eeg_channels)

for count, channel in enumerate(eeg_channels):
    print(count, channel)

"""

"""
#a = np.array([1, 2, 3, 4])
#print(a)
#np.save('a.npy', a)
from datetime import datetime

#a=np.load('test1.npy')
data=np.load('data2.npy')

print(data.shape)
data = data.transpose()
print(data.shape)

data2=np.unique(data)

print(data2.shape)


for e in data:
    #print(e[0])
    print( e[0], '-', datetime.fromtimestamp(e[22]) )


print(a[[0]])
print(a[[1]])
print(a[[2]])
print(a[[3]])
print(a[[4]])
print(a[[5]])

print(a[[6]])
print(a[[7]])
print(a[[8]])
print(a[[9]])
print(a[[10]])


print(a[[11]])
print(a[[12]])
print(a[[13]])
print(a[[14]])
print(a[[15]])


print(a[[16]])
print(a[[17]])
print(a[[18]])
print(a[[19]])
print(a[[20]])
print(a[[21]])
print(a[[22]])
"""

"""
timestamp=a[22][0]
dt_object = datetime.fromtimestamp(timestamp)
print("dt_object =", dt_object)
timestamp=a[22][210]
dt_object = datetime.fromtimestamp(timestamp)
print("dt_object =", dt_object)
"""