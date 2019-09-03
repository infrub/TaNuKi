import sys
sys.path.append('../../')
from tanuki import *
import numpy as np
import scipy as sp
import scipy.optimize as spo
import random
from colorama import Fore, Back, Style
import math



# test backward trunc houteisiki is correct
def test0210():
    b = 3
    n = b**b
    chi = b-1

    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate(A, chi=chi, memo=memo)
    B = M*S*N
    D = A - B

    ph = tensor_svd(B,"kl",svd_labels = "a")[2].truncate_index("a",2,3).remove_dummy_indices("a")
    p = ph.conjugate()
    assert B*p==zeros_tensor_like(B*p)

    u = A*p
    assert u==D*p

    pph = p.replace_labels("kr","pphout",inplace=False)[[]]*ph.replace_labels("kr","pphin",inplace=False)[[]]
    anti_pph = identity_tensor_like(pph) - pph

    def functor(G):
        return (G["kr"] * anti_pph["pphout"]).replace_labels("pphin", "kr", inplace=False) + u * ph

    assert A == functor(A)
    assert D == functor(D)
    assert B+u*ph == functor(B)

    def cost(G):
        return (H * functor(G)).norm()

    for _ in range(100):
        G = random_tensor_like(D)
        assert cost(G) > cost(D)

    niseu = random_tensor_like(u)
    assert abs(cost(u*ph) - cost(niseu*ph)) < 1e-5
    assert cost(niseu*ph) > cost(D)
    assert abs(cost(D) - cost(D+niseu*ph)) < 1e-5

    assert cost(A) > cost(D)
    assert cost(B) > cost(D)


    HETA = H
    HETAh = H.adjoint(["kl","kr"],style="aster")
    ETA = V
    HZETA = (anti_pph["pphin"]*H["kr"]).replace_labels("pphout","kr",inplace=False)
    HZETAh = HZETA.adjoint(["kl","kr"],style="aster")
    ZETA = HZETA * HZETAh
    HCV = HETA*u*ph

    assert np.linalg.matrix_rank(ZETA.to_matrix(["kl","kr"])) == b*b-b

    def costVector(G):
        return HZETA*G + HCV

    for _ in range(10):
        G = random_tensor_like(D)
        assert abs( costVector(G).norm() - cost(G) ) < 1e-5

    assert abs(costVector(D).norm() - cost(D)) < 1e-5

    def tensor_to_real_vector(G):
        G = G.to_vector(["kl","kr"])
        G = np.concatenate([np.real(G), np.imag(G)])
        return G

    def real_vector_to_tensor(G):
        G = G[:b*b] + 1.0j*G[b*b:]
        G = vector_to_tensor(G, (b,b), ["kl","kr"])
        return G

    def tensor_to_real_costVector(CV):
        CV = CV.to_vector(["extraction"])
        CV = np.concatenate([np.real(CV), np.imag(CV)])
        return CV





    def wrapped_costVector(G):
        G = real_vector_to_tensor(G)
        CV = costVector(G)
        return tensor_to_real_costVector(CV)

    #print(wrapped_costVector(D.to_vector(["kl","kr"])))


    G = random_tensor_like(D)
    G = tensor_to_real_vector(G)
    optG = spo.leastsq(wrapped_costVector, G)[0]
    optG = real_vector_to_tensor(optG)
    fnctOptG = functor(optG)
    assert abs(cost(optG) - cost(fnctOptG)) < 1e-5
    assert abs(cost(fnctOptG) - cost(D)) < 1e-5
    assert fnctOptG == D


    assert optG * ZETA + HCV*HZETAh 
    print(fnctOptG * ZETA + HCV*HZETAh)
    print(fnctOptG * HZETA * HETAh + HCV*HETAh)



# test backward opttrunc repeating doesn't give correct optimal_trunc
def test0211():
    b = 7
    n = b**b

    H = random_tensor((b,b,n),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    A = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    memo = {}
    M,S,N = ENV.optimal_truncate(A, chi=b-1, memo=memo)
    A1 = M*S*N
    print(memo["iter_times"])
    memo = {}
    M,S,N = ENV.optimal_truncate(A1, chi=b-2, memo=memo)
    A11 = M*S*N
    print(memo["iter_times"])
    memo = {}
    M,S,N = ENV.optimal_truncate(A, chi=b-2, memo=memo)
    A2 = M*S*N
    print(memo["iter_times"])

    print(A11-A2)



test0211()

