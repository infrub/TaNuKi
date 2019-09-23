import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from datetime import datetime

pd.options.display.max_columns = 30
pd.options.display.width = 160
MAX_PDF_PAGE = 100


def tuple_add(xs,ys):
    return tuple(x+y for (x,y) in zip(xs,ys))

def tuple_sub(xs,ys):
    return tuple(x-y for (x,y) in zip(xs,ys))

def tuple_mul(xs,k):
    return tuple(x*k for x in xs)

def tuple_div(xs,k):
    return tuple(x/k for x in xs)

def tuple_abs(xs):
    return math.sqrt(sum((x*x for x in xs)))

def shori_munxy():
    inpt = eval(open("epm033_work/munsell_xyY_raw.py").read())
    oupt = {mH*2.5:{} for mH in range(40)}
    for k,v in inpt.items():
        k0 = k[0]
        if k0[-2:] == "YR":
            H = float(k0[:-2])+10
        elif k0[-2:] == "GY":
            H = float(k0[:-2])+30
        elif k0[-2:] == "BG":
            H = float(k0[:-2])+50
        elif k0[-2:] == "PB":
            H = float(k0[:-2])+70
        elif k0[-2:] == "RP":
            H = float(k0[:-2])+90
        elif k0[-1:] == "R":
            H = float(k0[:-1])+0
        elif k0[-1:] == "Y":
            H = float(k0[:-1])+20
        elif k0[-1:] == "G":
            H = float(k0[:-1])+40
        elif k0[-1:] == "B":
            H = float(k0[:-1])+60
        elif k0[-1:] == "P":
            H = float(k0[:-1])+80
        if H == 100.0:
            H = 0.0
        oupt[H][(k[1],k[2])] = v

    open("epm033_work/munsell_xyY.py","w").write(str(oupt))



base_munsell_xyY_table = eval(open("epm033_work/munsell_xyY.py").read())

def musai_dounano():
    vs = [0.2,0.4,0.6,0.8] + list(range(1,11))
    table1 = {v:(0,0,0) for v in vs}
    table2 = {v:0 for v in vs}
    for mH in range(40):
        H = mH * 2.5
        for (V,C),xyY in base_munsell_xyY_table[H].items():
            if C == 2:
                table1[V] = tuple_add(table1[V], xyY)
                table2[V] = table2[V] + 1
    for v in vs:
        print(v)
        print(tuple_div(table1[v],table2[v]))






def base_munsell_to_xyY(H,V,C):
    if C == 0:
        Y = {0.2:0.237,0.4:0.467,0.6:0.670,0.8:0.943,1:1.210,2:3.126,3:6.550,4:12.00,5:19.77,6:30.03,7:43.06,8:59.10,9:78.66,10:102.57}[V]
        return (1/3, 1/3, Y)
    return base_munsell_xyY_table[H][(V,C)]

def find_exist_C(H,V,C):
    C = int(C/2)*2
    while C > 0:
        if (V,C) in base_munsell_xyY_table[H]:
            break
        C -= 2
    return C

def munsell_to_xyY(H,V,C):
    H1 = (math.floor(H/2.5)*2.5)%100
    H2 = (math.ceil(H/2.5)*2.5)%100

    if V <= 0.2:
        V1 = 0.2
        V2 = 0.2
    elif 0.2 < V < 1:
        V1 = math.floor(V/0.2)*0.2
        V2 = math.ceil(V/0.2)*0.2
    elif 1 <= V <= 10:
        V1 = math.floor(V)
        V2 = math.ceil(V)
    else:
        V1 = 10
        V2 = 10

    if H1 == H2 and V1 == V2:
        C11 = find_exist_C(H1,V1,C)
        return base_munsell_to_xyY(H1,V1,C11)

    C11 = find_exist_C(H1,V1,C)
    C22 = find_exist_C(H2,V2,C)

    d1 = tuple_abs(tuple_sub((H,V,C),(H1,V1,C11)))
    d2 = tuple_abs(tuple_sub((H,V,C),(H2,V2,C22)))

    return tuple_div(tuple_add(tuple_mul(base_munsell_to_xyY(H1,V1,C11),d2), tuple_mul(base_munsell_to_xyY(H2,V2,C22),d1)),d1+d2)


def xyY_to_XYZ(x,y,Y):
    if y == 0:
        y = 1
    X = Y/y*x
    Z = Y/y*(1-x-y)
    return (X,Y,Z)

def XYZ_to_rgb(X,Y,Z):
    R = 0.41847*X - 0.15866*Y - 0.082835*Z
    G = -0.091169*X + 0.25243*Y + 0.015708*Z
    B = 0.00092090*X - 0.0025498*Y + 0.17860*Z
    return (R,G,B)



"""
print(munsell_to_xyY(80,6,11))
print(XYZ_to_rgb(*(xyY_to_XYZ(*munsell_to_xyY(80,6,11)))))
print(XYZ_to_rgb(*(xyY_to_XYZ(*munsell_to_xyY(0,5,16)))))
print(XYZ_to_rgb(1/3,1/3,1/3))
"""


minr,maxr,ming,maxg,minb,maxb = 100,0,100,0,100,0
for H in base_munsell_xyY_table.keys():
    for VC, xyY in base_munsell_xyY_table[H].items():
        rgb = xyY_to_XYZ(*xyY)
        minr = min(minr,rgb[0])
        ming = min(ming,rgb[1])
        minb = min(minb,rgb[2])
        maxr = max(maxr,rgb[0])
        maxg = max(maxg,rgb[1])
        maxb = max(maxb,rgb[2])


print(minr,maxr,ming,maxg,minb,maxb )