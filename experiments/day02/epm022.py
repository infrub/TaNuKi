import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math



# test show iter_times
def test0220():
    b = 3
    n = b*b
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    #A = random_tensor((b,b),["kl","kr"])
    A = random_diagonalTensor((b,),["kl","kr"])
    A = A.real()
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])
    for chi in [2]: #range(1,b+1):
        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=3000, chi=chi, memo=memo)
        print(chi, memo["iter_times"], memo)

test0220()