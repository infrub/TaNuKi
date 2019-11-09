import sys,os
sys.path.append('../../')
from tanuki import *
from tanuki.onedim import *
from tanuki.twodim import *
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






def epm0800():
    class A:
        def __init__(self):
            pass
        def __getitem__(self,arg):
            print(arg)
            print(type(arg))

    a = A()
    a[0]
    a[0:"fun"]
    a[0:1:[2,4]]
    a[0:1:2,4]


def epm0801():
    b = 2
    J = 1.0
    A = ones_tensor((b,),["a"])
    B = ones_tensor((b,),["b"])
    L = dummy_diagonalTensor()
    R = dummy_diagonalTensor()
    U = dummy_diagonalTensor()
    D = dummy_diagonalTensor()
    PSI = twodim.Ptn2DCheckerBTPS(A,B,L,R,U,D, width_scale=2, height_scale=2)

    print(PSI)

    PSI.locally_canonize_all()

    print(PSI)

    H2 = zeros_tensor((2,2,2,2),["o0","o1","i0","i1"])
    H2.data[0,0,0,0] = 1
    H2.data[0,1,0,1] = -1
    H2.data[1,0,1,0] = -1
    H2.data[1,1,1,1] = 1
    H2 = Opn1DTMO(H2,[["o0"],["o1"]],[["i0"],["i1"]],is_hermite=True)
    EH2 = H2.exp(-J)

    PSI.apply(EH2,"L")

    print(PSI)


epm0801()