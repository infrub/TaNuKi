import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math
from matplotlib import pyplot as plt



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
    M,S,N = ENV.optimal_truncate(A, maxiter=100, chi=chi, memo=memo, algname="alg01")
    print(memo)
    print(M*S*N)

    memo = {}
    M,S,N = ENV.optimal_truncate(A, maxiter=10, chi=chi, memo=memo, algname="alg02")
    print(memo)
    print(M*S*N)



# test katamuki keisan!
def epm0231():
    b,chi = 20,10
    n = b*b
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    #algnames = ["alg01", "alg07", "alg04", "alg04'", "alg14"]
    #algnames = ["alg04", "alg14", "alg15"]
    algnames = ["alg01", "alg08", "alg04"]
    for algname in algnames:
        print(algname)
        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=1000, chi=chi, memo=memo, algname=algname)
        print(S)
        print(memo)
        #print("\n\n\n\n\n")
        print()
        lastM = M


# test katamuki keisan!
def epm0231():
    b,chi = 10,8
    n = b*b
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    plt.figure()
    #algnames = ["alg01", "alg07", "alg04", "alg04'", "alg14"]
    #algnames = ["alg04", "alg14", "alg15"]
    algnames = ["alg08", "alg01", "alg04"]
    trueError = None
    for algname in algnames:
        print(algname)
        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=400, chi=chi, memo=memo, algname=algname)
        if trueError is None:
            trueError = memo["sq_diff"]*(1-1e-7)
        plt.plot(np.array(memo.pop("fxs"))-trueError, label=algname)
        print(S)
        print(memo)
        print()

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()




epm0231()