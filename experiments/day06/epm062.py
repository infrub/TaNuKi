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

pd.options.display.max_columns = 30
pd.options.display.width = 160
np.set_printoptions(linewidth=float("inf"))
tnc.display_max_size = float("inf")



def partition_function(beta, Jx, Jy, Lx, Ly):
    a = beta * Jx
    b = beta * Jy
    gamma = [None for _ in range(2*Lx)]
    for k in range(2*Lx):
        cosh_g = ( ef_cosh(2*a) * ef_cosh(2*b) - cos(pi*k/Lx) * ef_sinh(2*b) ) / ef_sinh(2*a)
        gamma[k] = (cosh_g + (cosh_g * cosh_g - 1).sqrt()).log

    if ef_sinh(2*a) * ef_sinh(2*b) > 1: gamma[0] = -gamma[0]

    p0,p1,p2,p3 = 1.0,1.0,1.0,1.0
    for k in range(1,Lx+1):
        p0 *= 2 * ef_cosh(Ly * gamma[2*k-1] / 2)
        p1 *= 2 * ef_sinh(Ly * gamma[2*k-1] / 2)
        p2 *= 2 * ef_cosh(Ly * gamma[2*k-2] / 2)
        p3 *= 2 * ef_sinh(Ly * gamma[2*k-2] / 2)

    z = 0.5 * ( (2 * ef_sinh(2*a)) ** (Lx*Ly/2) ) * (p0 + p1 + p2 - p3);
    return z





beta = 1.0
J = 0.5
width_scale = 2
height_scale = 2
chi = 4

print(f"beta:{beta}, width_scale:{width_scale}, height_scale:{height_scale}, chi:{chi}\n\n")


def epm0620_core(symbol):
    #push# make Ising model partition function TPK
    gate = zeros_tensor((2,2,2,2), ["ain","aout","bin","bout"])
    gate.data[1,1,1,1] = np.exp(beta*J)
    gate.data[0,0,0,0] = np.exp(beta*J)
    gate.data[0,0,1,1] = np.exp(-beta*J)
    gate.data[1,1,0,0] = np.exp(-beta*J)
    gate = onedim.Opn1DTMO(gate, [["aout"],["bout"]], [["ain"],["bin"]])
    A = identity_tensor((2,), labels=["ain","aout"])
    B = identity_tensor((2,), labels=["bin","bout"])

    Ss = []
    for _ in range(4):
        funi = gate.to_BTPO()
        a,S,b = funi.tensors[0], funi.bdts[1], funi.tensors[1]
        A = A["aout"]*a["ain"]
        Ss.append(S)
        B = B["bout"]*b["bin"]
    L,R,U,D = tuple(Ss)
    A = A.trace("aout","ain")
    B = B.trace("bout","bin")

    Z_TPK = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)


    def calc_Z(symbol):
        if symbol == "othn":
            return partition_function(beta,J,J,2**(width_scale),2**(height_scale))
        a,b,c,d = symbol[0],symbol[1],symbol[2],symbol[3]
        kwargs = {}
        kwargs["loop_truncation_algname"] = {"N":"naive","C":"canonize","I":"iterative"}[a]
        kwargs["env_choice"] = {"N":"no","H":"half"}[b]
        kwargs["contract_before_truncate"] = {"A":False,"B":True}[c]
        kwargs["drill_parity"] = {"E":0,"O":1}[d]
        return Z_TPK.calculate(chi=chi, **kwargs)

    #@timeout(120)
    def calc_F_value(symbol):
        Z = calc_Z(symbol)
        return -1.0 / beta * Z.log

    return calc_F_value(symbol)



def epm0620():
    #symbols = ["othn"] + [a+b+c+d for a in "NCI" for b in "HN" for c in "AB" for d in "EO"]
    #symbols = ["othn"] + [a+bc+d for a in "NCI" for bc in ["HA","HB","NB"] for d in "EO"]
    symbols = ["othn"] + [a+bc+d for a in "I" for bc in ["NA"] for d in "EO"]

    results = []
    for symbol in symbols:
        #if kwargs!="othn" and kwargs["loop_truncation_algname"] == "canonize": continue
        print()
        print(symbol)
        try:
            F_value = epm0620_core(symbol)
            print(symbol, F_value)
            results.append((symbol,F_value))
        except Exception as e:
            print(symbol, e)
            results.append((symbol,9999))
            raise e

    print("\n\n")

    results.sort(key=lambda a: a[1])
    for symbol, F_value in results:
        print(symbol, F_value)





#epm0620_core("INBO")
epm0620()

