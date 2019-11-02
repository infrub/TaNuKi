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







beta = 1.0
J = 0.9
width_scale = 5
height_scale = 5
chi = 10

print(f"beta:{beta}, width_scale:{width_scale}, height_scale:{height_scale}, chi:{chi}\n\n")




def make_Z_TPS():
    gate = zeros_tensor((2,2,2,2), ["ain","aout","bin","bout"])
    gate.data[1,1,1,1] = np.exp(beta*J)
    gate.data[0,0,0,0] = np.exp(beta*J)
    gate.data[0,0,1,1] = np.exp(-beta*J)
    gate.data[1,1,0,0] = np.exp(-beta*J)
    gate = onedim.Opn1DTMO(gate, [["aout"],["bout"]], [["ain"],["bin"]])
    A = ones_tensor((2,), labels=["aout"])
    B = ones_tensor((2,), labels=["bout"])

    Ss = []
    for _ in range(4):
        funi = gate.to_BTPO()
        a,S,b = funi.tensors[0], funi.bdts[1], funi.tensors[1]
        A = A["aout"]*a["ain"]
        Ss.append(S)
        B = B["bout"]*b["bin"]
    L,R,U,D = tuple(Ss)
    A.replace_labels("aout","a")
    B.replace_labels("bout","b")

    return twodim.Ptn2DCheckerBTPS(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)


def make_random_TPS():
    b = 2
    chi = 20
    A = random_tensor((b,chi,chi,chi,chi),["a","al","ar","au","ad"])
    B = random_tensor((b,chi,chi,chi,chi),["b","bl","br","bu","bd"])
    L = random_diagonalTensor((chi,),["al","bl"])
    R = random_diagonalTensor((chi,),["ar","br"])
    U = random_diagonalTensor((chi,),["au","bu"])
    D = random_diagonalTensor((chi,),["ad","bd"])

    return twodim.Ptn2DCheckerBTPS(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)


def epm0700():
    Z = make_Z_TPS()
    print("ALURDB",Z.A*Z.L*Z.R*Z.U*Z.D*Z.B)
    Z.super_orthogonalize_ver1(maxiter=1)
    print("ALURDB",Z.A*Z.L*Z.R*Z.U*Z.D*Z.B)
    print("ALRU",(Z.A*Z.L*Z.R*Z.U).is_prop_right_semi_unitary(rows=Z.D)) #True(factor=1.0)


def epm0701():
    Z = make_random_TPS()
    print("ALURDB",Z.A*Z.L*Z.R*Z.U*Z.D*Z.B)
    Z.super_orthogonalize()
    print("ALURDB",Z.A*Z.L*Z.R*Z.U*Z.D*Z.B)
    print("ALRU",(Z.A*Z.L*Z.R*Z.U).is_prop_right_semi_unitary(rows=Z.D)) #True(factor=1.0)


epm0701()