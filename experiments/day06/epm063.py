import sys,os
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
import textwrap
from timeout_decorator import timeout, TimeoutError
from math import *
import itertools

pd.options.display.max_columns = 30
pd.options.display.width = 160
np.set_printoptions(linewidth=float("inf"))
tnc.display_max_size = float("inf")



b = 2
chi = None
width_scale = 2
height_scale = 2
A = random_tensor((b,b,b,b),["al","ar","au","ad"])
B = random_tensor((b,b,b,b),["bl","br","bu","bd"])
L = random_diagonalTensor((b,),["al","bl"])
R = random_diagonalTensor((b,),["ar","br"])
U = random_diagonalTensor((b,),["au","bu"])
D = random_diagonalTensor((b,),["ad","bd"])


def epm0630_core(symbol):
    Z_TPK = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)

    if symbol == "othn":
        return partition_function(beta,J,J,2**(width_scale),2**(height_scale))
    a,b,c,d = symbol[0],symbol[1],symbol[2],symbol[3]
    kwargs = {}
    kwargs["loop_truncation_algname"] = {"N":"naive","C":"canonize","I":"iterative"}[a]
    kwargs["env_choice"] = {"N":"no","H":"half"}[b]
    kwargs["contract_before_truncate"] = {"A":False,"B":True}[c]
    kwargs["drill_parity"] = {"E":0,"O":1}[d]
    return Z_TPK.calculate(chi=chi, **kwargs)


def epm0630():
    #symbols = [a+b+c+d for a in "NCI" for b in "HN" for c in "AB" for d in "EO"]
    symbols = [a+b+c+d for a in "NCI" for b in "HN" for c in "A" for d in "EO"]

    results = []
    for symbol in symbols:
        print()
        print(symbol)
        try:
            F_value = epm0630_core(symbol)
            print(symbol, F_value)
            results.append((symbol,F_value))
        except Exception as e:
            print(symbol, e)
            results.append((symbol,9999))
            raise e

    print("\n\n")

    results.sort(key=lambda a: abs(a[1]))
    for symbol, F_value in results:
        print(symbol, F_value)




epm0630()

