import sys
sys.path.append('../../')
from tanuki import *
import numpy as np

def test0100():
    b = 10
    H_L = random_tensor((b,20),["kl","extraction"])
    V_L = H_L * H_L.adjoint("kl",style="aster")
    #print(V_L)
    H_R = random_tensor((b,20),["kr","extraction"])
    V_R = H_R * H_R.adjoint("kr",style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.BridgeBondEnv(V_L, V_R, ["kl"], ["kr"])

    def wa(chi):
        M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
        #print(chi, sigma0 - M*S*N)
        #print(S)
        print(S)

    for i in range(1,11):
        wa(i)

def test0101():
    b = 5
    H = random_tensor((b,b,b*b+5),["kl","kr","extraction"])
    V = H * H.adjoint(["kl","kr"],style="aster")
    sigma0 = random_tensor((b,b),["kl","kr"])
    ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

    
    for chi in range(1,b+1):
        M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
        #print(M)
        #print(S)
        #print( (sigma0 - M*S*N).norm() )
        print(M*S*N*H)
    """
    M1,S1,N1 = ENV.optimal_truncate(sigma0, chi=1)
    M2,S2,N2 = ENV.optimal_truncate(sigma0, chi=2)
    m10 = M1.data[:,0:1]
    m20 = M2.data[:,0:1]
    m21 = M2.data[:,1:2]
    m1 = M1.data[:2,0]
    m2 = M2.data[:2,:2]
    print(m1,m2)
    ab = np.linalg.solve(m2,m1)
    print(ab)
    a = ab[0]
    b = ab[1]
    print(a * m20 + b * m21)
    print(m10)
    """

def test0102():
    def f(b,n,chi):
        H = random_tensor((b,b,n),["kl","kr","extraction"])
        V = H * H.adjoint(["kl","kr"],style="aster")
        sigma0 = random_tensor((b,b),["kl","kr"])
        ENV = bondenv.UnbridgeBondEnv(V, ["kl"], ["kr"])

        M,S,N = ENV.optimal_truncate(sigma0, chi=chi)
        print((sigma0-M*S*N)*H)

    f(5,5,1)



test0102()