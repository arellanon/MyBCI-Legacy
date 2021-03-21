#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 21:37:02 2021

@author: nahuel
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 20:39:10 2021

@author: nahuel
"""
import os
from os import listdir
from os.path import isfile, isdir


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

def new_name(path, inicial):
    #Calculamos directorio enumerado    
    directorios = ls1(path)
    #filtramos los directorios que empiezan con T
    lista = filtro(directorios, inicial)
    num=1
    if lista:
        num = max(lista) + 1
    name = inicial + str(num)
    return name