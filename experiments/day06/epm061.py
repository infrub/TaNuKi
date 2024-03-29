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



def epm0610():
    A_ = onedim.random_cyc1DBTPS(["p0","p1","p2","p3"], virt_labelss=["v0","v1","v2","v3"], chi=6, phys_dim=2)

    A = A_
    print(A.to_TMS().tensor)

    A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
    w = A.truncate(chi=5, algname="iterative")
    print(A.to_TMS().tensor*w)

    A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
    w = A.truncate(chi=4, algname="iterative")
    print(A.to_TMS().tensor*w)

    A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
    w = A.truncate(chi=5, algname="canonize")
    print(A.to_TMS().tensor*w)



#print(epm0610())


def epm0611():
    A = random_tensor((2,3),["a","b"])
    Q,R = A.qr("a", force_diagonal_elements_positive=True)
    print(Q)
    print(Q.is_unitary("a"))
    print(R)
    print(R.is_tril("b"))
    print(A==Q*R)





def epm0612():
    A = onedim.random_cyc1DBTPS(["p0","p1","p2","p3"], virt_labelss=["v0","v1","v2","v3"], chi=6, phys_dim=2)

    print("\nLeft half transfer eigen:")
    memo={}
    print(A.get_left_half_transfer_eigen(memo=memo))
    print(memo)

    print("\nRight half transfer eigen:")
    memo={}
    print(A.get_right_half_transfer_eigen(memo=memo))
    print(memo)

    print("\nuniversally_canonize:")
    print(A.to_TMS().tensor)
    w = A.universally_canonize_around_end_bond()
    print(A.to_TMS().tensor*w)




def epm0613():
    A_ = onedim.random_cyc1DBTPS(["p0","p1","p2","p3"], virt_labelss=["v0","v1","v2","v3"], chi=6, phys_dim=2)

    A = A_
    print(A.to_TMS().tensor)

    for chi in [6,5,4,3,2,1]:
        print(f"chi={chi}")
        A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
        w = A.universally_canonize_around_end_bond(chi=chi)
        print(A.to_TMS().tensor*w, A.bdts[0])





def epm0614():
    A_ = onedim.random_cyc1DBTPS(["p0","p1","p2","p3"], virt_labelss=["v0","v1","v2","v3"], chi=6, phys_dim=2)

    A = A_
    print(A.to_TMS().tensor)

    for _ in range(1):
        A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
        w = A.universally_canonize_around_end_bond()
        print(A.bdts[0])

    for _ in range(1):
        A = onedim.Cyc1DBTPS(A_.tensors, A_.bdts)
        w = A.universally_canonize_around_end_bond_ver1()
        print(A.bdts[0])


epm0610()