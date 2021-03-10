#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 20:25:04 2021

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

from datetime import datetime

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds

#a=np.load('test1.npy')

index_ts=np.load('index_ts.npy')
print(index_ts.shape)
#print(data[0:256])


"""
print( datetime.fromtimestamp(index_ts[0]))
print( datetime.fromtimestamp(index_ts[10]))
print( datetime.fromtimestamp(index_ts[20]))
"""

totalData=np.load('totalData.npy')
print(totalData.shape)
#print(data[0:256])


labels=np.load('labels.npy')
print(labels.shape)
print(labels)
#print(labels[1][0])

#print(type(index_ts[0]))


#index_ts = index_ts[:2000]


#1720
#0.023936748504638672
#0.023936748504638672

events=np.load('events.npy')
print(events.shape)
print(events)
np.save('DATA\labels.npy', labels)

"""
y=labels[0][1]
ind=0
for x in index_ts:
    #print( datetime.fromtimestamp(x) )
    print(ind,": ", x, y, " - ", abs(x-y) ) 
    ind=ind+1


labels=labels.transpose()
print(labels.shape)
posiciones=None

for x in labels:
    resta = abs(index_ts - x[0])
    pos = np.where(min(resta) == resta)[0]
    if posiciones is None:
        posiciones = pos
    else:
        posiciones = np.append(posiciones, pos)
        
        
events = np.zeros((len(labels) , 3), int)
events[:, 0] = posiciones.astype(int)
events[:, 2] = labels[:,1].astype(int)

print(events)

ind=0
for x in labels[0]:
    resta = abs(index_ts - x)
    pos = np.where(min(resta) == resta)[0]
    print(pos)
    #print(ind, datetime.fromtimestamp(x), ": ", index_ts[ pos[0][0] ] )
    #print(ind, x, ": ", index_ts[ pos[0][0] ] , pos[0][0], labels[1][ind])
    ind=ind+1
  
"""

"""
np_array = np.array((1, 5, 9, 3, 7, 2, 0))
print( np.where(min(np_array) == np_array) )


ind=0
for x in labels[0]:
    #print( datetime.fromtimestamp(x) )
    print(type(x))
    print( np.where(index_ts == x) )
    ind=ind+1


print( datetime.fromtimestamp(labels[0][0]))
print( datetime.fromtimestamp(labels[0][1]))
print( datetime.fromtimestamp(labels[0][2]))
print( datetime.fromtimestamp(labels[0][3]))


np_array = np.array((1, 5, 9, 3, 7, 2, 0))
print( np.where(np_array == 5) )
"""