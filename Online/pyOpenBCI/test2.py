#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 22:35:31 2021

@author: nahuel
"""
import numpy as np

#a = np.array([1, 2, 3, 4])
#print(a)
#np.save('a.npy', a)
from datetime import datetime

#a=np.load('test1.npy')
data=np.load('data1.npy')

print(data.shape)
data = data.transpose()
print(data.shape)
for e in data:
    #print(e[0])
    print( e[0], '-', datetime.fromtimestamp(e[22]) )

"""
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