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

    funi = twodim.Prd2DCheckerTPK(A,B,L,R,U,D, scale=2)
    print(funi)
    for chi in [2,3,4,5]:
        print(chi, funi.calculate(chi=chi))


epm0600()