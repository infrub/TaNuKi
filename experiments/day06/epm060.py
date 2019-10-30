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




def epm0603():
    beta = 1.0
    J = 0.5
    width_scale = 4
    height_scale = 4
    chi = 8

    print(f"beta:{beta}, width_scale:{width_scale}, height_scale:{height_scale}, chi:{chi}\n\n")

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
    #Z_TPK = twodim.Ptn2DRhombusBTPK(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)
    #pop# make Ising model TPK

    #symbols = ["otehon"] + [a+b+c+d for a in "NCRI" for b in "HN" for c in "AB" for d in "EO"]
    #kwargss = ["otehon"] + [dict(loop_truncation_algname=loop_truncation_algname, initial_value=initial_value, env_choice=env_choice, contract_before_truncate=contract_before_truncate, drill_parity=drill_parity) for loop_truncation_algname, initial_value in [("naive",None), ("canonize",None), ("iterative","random"), ("iterative", "naive_truncation")] for env_choice in ["half","no"] for contract_before_truncate in [False,True] for drill_parity in [0,1]]

    symbols = ["otehon"] + [a+b+c+d for a in "NCI" for b in "HN" for c in "AB" for d in "EO"]
    kwargss = ["otehon"] + [dict(loop_truncation_algname=loop_truncation_algname, env_choice=env_choice, contract_before_truncate=contract_before_truncate, drill_parity=drill_parity) for loop_truncation_algname in ["naive", "canonize", "iterative"] for env_choice in ["half","no"] for contract_before_truncate in [False,True] for drill_parity in [0,1]]


    results = []
    for symbol, kwargs in zip(symbols,kwargss):
        #if kwargs!="otehon" and kwargs["loop_truncation_algname"] == "canonize": continue
        print()
        print(symbol)
        try:
            @timeout(20)
            def funi():
                if kwargs == "otehon":
                    return partition_function(beta,J,J,2**(width_scale),2**(height_scale))
                else:
                    return Z_TPK.calculate(chi=chi, **kwargs)
            Z = funi()
            F_value = -1.0 / beta * Z.log
            print(symbol, Z, F_value)
            results.append((symbol,F_value))
        except Exception as e:
            print(symbol, e)
            results.append((symbol,114514))
            #raise e

    print("\n\n")

    results.sort(key=lambda a: a[1])
    for symbol, F_value in results:
        print(symbol, F_value)





epm0603()






def epm0604():
    chi = 8

    def calc_with_tn(beta, J, width_scale, height_scale):
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
        #Z_TPK = twodim.Ptn2DRhombusBTPK(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)
        #pop# make Ising model TPK

        kwargs = dict(env_choice="half", contract_before_truncate=False, loop_truncation_algname="naive", drill_parity=0)
        
        Z = Z_TPK.calculate(chi=chi, **kwargs)
        F_value = -1.0 / beta * Z.log
        return F_value


    def calc_with_otehon(beta, J, width_scale, height_scale):
        Z = partition_function(beta,J,J,2**(width_scale),2**(height_scale))
        F_value = -1.0 / beta * Z.log
        return F_value



    beta = 0.1
    J = 1.0

    for width_scale in range(6):
        for height_scale in range(6):
            print(calc_with_otehon(beta, J, width_scale, height_scale), end=" | ")
        print()

    print()
    for width_scale in range(1,7):
        for height_scale in range(1,7):
            print(calc_with_tn(beta, J, width_scale, height_scale), end=" | ")
        print()





def epm0605():
    beta = 0.5
    J = 1.0
    chi = 8


    print("Checker1 Otehon")
    for _ in range(1):
        Z = partition_function(beta,J,J,2,1)
        F_value = -1.0 / beta * Z.log
        print(F_value)


    print("Checker1 Tegaki")
    for _ in range(1):
        gate = zeros_tensor((2,2,2,2), ["ain","aout","bin","bout"])
        gate.data[1,1,1,1] = np.exp(beta*J)
        gate.data[0,0,0,0] = np.exp(beta*J)
        gate.data[0,0,1,1] = np.exp(-beta*J)
        gate.data[1,1,0,0] = np.exp(-beta*J)
        A = identity_tensor((2,), labels=["ain","aout"])
        B = identity_tensor((2,), labels=["bin","bout"])
        G = A * B

        G = G["aout","bout"]*gate["ain","bin"]
        G = G["aout","bout"]*gate["ain","bin"]
        #print(gate.trace("bout","ain").replace_labels("bin","ain", inplace=False), np.exp(beta*J))
        G = G["aout"] * gate.trace("bout","ain").replace_labels("bin","ain", inplace=False)["ain"]
        G = G["bout"] * gate.trace("bout","ain").replace_labels("aout","bout", inplace=False)["bin"]
        G = G.trace(["aout","bout"],["ain","bin"])

        Z = ExpFloat(G.to_scalar())
        F_value = -1.0 / beta * Z.log
        print(F_value)


    print("Checker2 Otehon")
    for _ in range(1):
        Z = partition_function(beta,J,J,2,2)
        F_value = -1.0 / beta * Z.log
        print(F_value)


    print("Checker2")
    for _ in range(1):
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

        Z_TPK = twodim.Ptn2DCheckerBTPK(A,B,L,R,U,D, width_scale=1, height_scale=1)

        kwargs = dict(env_choice="half", contract_before_truncate=False, loop_truncation_algname="naive", drill_parity=0)
        
        Z = Z_TPK.calculate(chi=chi, **kwargs)
        F_value = -1.0 / beta * Z.log
        print(F_value)





    print("Rhombus Otehon")
    for _ in range(1):
        F_value = -1.0 / beta * np.log(4*np.cosh(4*beta))
        print(F_value)



    print("Rhombus Tegaki")
    for _ in range(1):
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

        Z = ExpFloat((A*L*R*U*D*B).to_scalar())
        F_value = -1.0 / beta * Z.log
        print(F_value)


    print("Rhombus")
    for _ in range(1):
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

        Z_TPK = twodim.Ptn2DRhombusBTPK(A,B,L,R,U,D, width_scale=width_scale, height_scale=height_scale)
        #pop# make Ising model TPK

        kwargs = dict(env_choice="half", contract_before_truncate=False, loop_truncation_algname="naive", drill_parity=0)
        
        Z = Z_TPK.calculate(chi=chi, **kwargs)
        F_value = -1.0 / beta * Z.log
        print(F_value)



