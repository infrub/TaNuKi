import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import random
from colorama import Fore, Back, Style
import math



def test0200():
    b = 4
    n = b**b
    chi = b-1

    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate(sigma0, chi=chi, memo=memo)
    print(S)
    print(tensor_svd(sigma0, ["kl"])[1], sep="\n")
    diff = ((M*S*N)-sigma0)
    print(tensor_svd(diff, ["kl"])[1], sep="\n")
    er1 = diff.norm()
    er2 = (diff*H).norm()
    print(diff)
    print(np.linalg.matrix_rank(diff.data))
    print(er1)
    print(er2)
    print(memo)


test0200()