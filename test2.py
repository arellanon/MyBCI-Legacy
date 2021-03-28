#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 02:03:42 2021

@author: nahuel
"""
import numpy as np

#events = np.zeros((1, 8, 251), float)
#print(type(events[0][0][0]))
#events[:, 0] = posiciones.astype(int)
#events[:, 2] = labels[:,1].astype(int)

a = np.arange(8)
# arrayA = array([0, 1, 2, 3, 4, 5, 6, 7])

b= np.resize(a, (1, 8, 251))
#array([[0, 1, 2, 3],
#       [4, 5, 6, 7]])
print(a)
print(b.shape)