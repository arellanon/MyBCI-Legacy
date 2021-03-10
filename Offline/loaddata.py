# -*- coding: utf-8 -*-
"""
@Time    : 202/08/12
@Author  : arellanon
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
import os

import mne


def creatEventsArray(fp):
    """Cargamos informacion sobre los eventos"""
    data_mrk_txt = np.loadtxt(fp['mrk'])
    true_labels = np.loadtxt(fp['lab']).astype('int')
    events = np.zeros((len(true_labels) , 3), int)
    events[:, 0] = data_mrk_txt[:,0].astype(int)
    events[:, 2] = true_labels
    return events


def creatRawArray(fp):
    freq=fp['freq']
    #data_cnt_txt = np.loadtxt(fp['cnt'])
    #data_cnt = data_cnt_txt.transpose()
    
    data_cnt= np.load(fp['cnt'])
    print(data_cnt.shape)
    
    ch_names_txt = open(fp['chn'], "r")
    ch_names = ch_names_txt.read().split(',')
    for i in range(len(ch_names)):
        ch_names[i]=ch_names[i].strip()
    info = mne.create_info(ch_names, freq, 'eeg')
    print('dimensiones data_cnt: ', data_cnt.shape)
    raw = mne.io.RawArray(data_cnt, info, first_samp=0, copy='auto', verbose=None)
    return raw