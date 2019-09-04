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
import matplotlib.cm as cm


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




# test SOR-like kasoku param
def epm0232():
    b,chi = 10,8
    n = b*b
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    plt.figure()

    memo = {}
    M,S,N = ENV.optimal_truncate(A, maxiter=1000, conv_atol=1e-14, conv_rtol=1e-14, chi=chi, memo=memo, algname="alg04")
    trueError = memo["sq_diff"]*(1-1e-7)

    def yaru(algname, kasoku, color):
        print(f"{algname}({kasoku:4.2f})")
        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=400, chi=chi, memo=memo, algname=algname, kasoku=kasoku)
        y_X = np.array(memo.pop("fxs"))
        plt.plot(y_X-trueError, label=f"{algname}({kasoku:4.2f})", color=color)
        print(S)
        print(memo)
        print()

    yaru("alg04", 1, "black")

    for kasoku in np.linspace(1.0,1.9,10):
        yaru("alg08", kasoku, cm.gist_rainbow(float(kasoku-1.65)*3))

    yaru("alg01", 1, "gray")
    

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()




# motto kuwasiku
def epm0233():
    b,chi = 10,8
    n = b*b
    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    plt.figure()

    memo = {}
    M,S,N = ENV.optimal_truncate(A, maxiter=1000, conv_atol=1e-14, conv_rtol=1e-14, chi=chi, memo=memo, algname="alg04")
    trueError = memo["sq_diff"]*(1-1e-10)

    def yaru(algname, kasoku, color):
        print(f"{algname}({kasoku:4.2f})")
        memo = {}
        M,S,N = ENV.optimal_truncate(A, maxiter=400, chi=chi, memo=memo, algname=algname, kasoku=kasoku)
        y_X = np.array(memo.pop("fxs"))
        plt.plot(y_X-trueError, label=f"{algname}({kasoku:4.2f})", color=color)
        print(S)
        print(memo)
        print()

    yaru("alg04", 1, "black")

    for kasoku in np.linspace(1.65,1.95,11):
        yaru("alg08", kasoku, cm.gist_rainbow(float(kasoku-1.65)*3))

    yaru("alg01", 1, "gray")
    

    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()


epm0233()