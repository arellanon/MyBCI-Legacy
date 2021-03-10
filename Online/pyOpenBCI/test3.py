#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:01:22 2021

@author: nahuel
"""
import time
import random
import os
from datetime import datetime

def test():
    run_n = 1
    trial_per_run = 4
    time_trial = 7
    time_pause = 0
    
    #archivo labels
    fout=open("labels.txt","w")
    fout.close()
    
    #variables para sonido beep
    duration = 1  # seconds
    freq = 440  # Hz
    
    stack = []
    left  = [0] * (trial_per_run // 2)
    rigth = [1] * (trial_per_run // 2)    
    stack = left + rigth
    print(stack)
    random.shuffle(stack)
    print(stack)
    
    fout=open("output.txt","a")
    for i in range(run_n):
        print('Corrida N#: ', run_n)
        for x in stack:
            ts = time.time()
            print()
            time.sleep(time_pause)
            print(x, ' ', ts, ' - ', datetime.fromtimestamp(ts))
            fout.write(str(x)+ "\t" + str(ts) + "\n")
            os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            for j in range(time_trial):
                if x == 0:
                    print('<', end="")
                else:
                    print('>', end="")
                time.sleep(1)
    fout.close()
    

if __name__ == "__main__":
    test()    