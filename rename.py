#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 03:18:23 2021

@author: nahuel
"""
import numpy as np
path_raiz = 'DATA/'
name = 'T12'
path = path_raiz + name
data = np.load(path + '/data.npy')
print(data.shape)
data = data.transpose()
print(data.shape)
np.save(path + '/data.npy', data)
