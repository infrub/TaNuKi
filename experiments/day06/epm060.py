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

pd.options.display.max_columns = 30
pd.options.display.width = 160
np.set_printoptions(linewidth=float("inf"))
tnc.display_max_size = float("inf")



def epm0600():
    b = 2
    A = random_tensor((b,b,b,b),["AL","AR","AU","AD"])
    B = random_tensor((b,b,b,b),["BL","BR","BU","BD"])
    L = random_diagonalTensor((b,),["AL","BL"])
    R = random_diagonalTensor((b,),["AR","BR"])
    U = random_diagonalTensor((b,),["AU","BU"])
    D = random_diagonalTensor((b,),["AD","BD"])

    funi = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, scale=2)
    print(funi)
    for chi in [2,3,4,5]:
        print(chi, funi.calculate(chi=chi))


def epm0601():
    b = 2
    A = random_tensor((b,b,b,b),["AL","AR","AU","AD"])
    B = random_tensor((b,b,b,b),["BL","BR","BU","BD"])
    L = random_diagonalTensor((b,),["AL","BL"])
    R = random_diagonalTensor((b,),["AR","BR"])
    U = random_diagonalTensor((b,),["AU","BU"])
    D = random_diagonalTensor((b,),["AD","BD"])

    funi = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, scale=3)
    print(funi)
    for algname in ["LN","YGW"]:
        for chi in [3,4,5,6,7]:
            #print(algname, chi, funi.calculate(chi=chi, algname=algname))
            print(algname, chi, complex(funi.calculate(chi=chi, algname=algname)), funi.calculate(chi=chi, normalize=False, algname=algname))


def epm0602():
    b = 2
    A = random_tensor((b,b,b,b),["AL","AR","AU","AD"])
    B = random_tensor((b,b,b,b),["BL","BR","BU","BD"])
    L = random_diagonalTensor((b,),["AL","BL"])
    R = random_diagonalTensor((b,),["AR","BR"])
    U = random_diagonalTensor((b,),["AU","BU"])
    D = random_diagonalTensor((b,),["AD","BD"])

    funi = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, scale=3)
    print(funi)
    for chi in [4,7,10,16,17]:
        for algname in ["LN","YGW2","YGW1"]:
            #print(algname, chi, funi.calculate(chi=chi, algname=algname))
            print(algname, chi, complex(funi.calculate(chi=chi, algname=algname)))



def epm0603():
    beta = 0.1
    scale = 17

    #push# make Ising model partition function TPK
    gate = zeros_tensor((2,2,2,2), ["ain","aout","bin","bout"])
    gate.data[1,1,1,1] = np.exp(beta)
    gate.data[0,0,0,0] = np.exp(beta)
    gate.data[0,0,1,1] = np.exp(-beta)
    gate.data[1,1,0,0] = np.exp(-beta)
    gate = onedim.Obc1DTMO(gate, [["aout"],["bout"]], [["ain"],["bin"]])
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

    Z = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, scale=scale)
    #pop# make Ising model TPK

    for chi in [16]:
        for algname in ["LN","YGW2"]:
            re = Z.calculate(chi=chi, algname=algname)
            #Z_value = float(re)
            F_value = -1.0 / beta * re.log
            print(algname, chi, F_value)







epm0603()