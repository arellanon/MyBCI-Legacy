#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:01:22 2021

@author: nahuel
"""
import time
import random
import os

def main ():

    run_n = 1
    trial_per_run = 14
    time_trial = 7
    
    duration = 1  # seconds
    freq = 440  # Hz
    
    stack = []
    left  = ['<-'] * (trial_per_run // 2)
    rigth = ['->'] * (trial_per_run // 2)    
    stack = left + rigth
    print(stack)
    random.shuffle(stack)    
    print(stack)
    
    for i in range(run_n):
        print('Corrida N#: ', run_n)
        for x in stack:
            print('next')
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            for j in range(time_trial):
                print(x, end="")
                time.sleep(1)

if __name__ == "__main__":
    main()