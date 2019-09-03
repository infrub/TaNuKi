import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math



# test koukahou katamukikeisannnashi
def epm0230():
    b = 4
    n = b*b
    chi = b-1
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate_alg01(A, maxiter=100, chi=chi, memo=memo)
    print(memo)
    print(M*S*N)

    memo = {}
    M,S,N = ENV.optimal_truncate_alg02(A, maxiter=1000, chi=chi, memo=memo)
    print(memo)
    print(M*S*N)



test0230()