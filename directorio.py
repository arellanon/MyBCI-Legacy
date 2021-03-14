# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
from os import listdir
from os.path import isfile, isdir


path='DATA/'
name='T1'

def ls1(path):
    lista = []
    if os.path.exists(path) and os.path.isdir(path):
        lista = [obj for obj in listdir(path) if isdir(path + obj)]
    return lista

def filtro(lista, inicial):
    result=[]
    for a in lista:
        if a[:len(inicial)]==inicial:
            result.append( int(a[1:]) )
    return result

if __name__ == "__main__":
    directorios = ls1(path)
    inicial="T"
    #filtramos los directorios que empiezan con T
    lista = filtro(directorios, inicial)
    if lista:
        num = max(lista) + 1
        name = inicial + str(num)    
    new =path + name
    print(new)
    #os.makedirs(new, exist_ok=True)