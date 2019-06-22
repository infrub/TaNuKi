import sys
sys.path.append('../../')
from tanuki import *
from tanuki.matrices import *
import numpy as np

def test0060():
    A = random_tensor((4,5,6),["a","b","c"])
    B = random_tensor_like(A)
    print(A)
    print(B)
    print(A+B)
    print(A["a"]*B["a"])

def test0061():
    A = random_tensor((2,3,4),["a","b","c"])
    print(A)
    U, S, V = tensor_svd(A, "a")
    print(U,S,V)
    B = U*S*V
    print(B)
    print(A==B)

def test0062():
    A = random_tensor((5,5))
    print(A)
    B = A.to_diagonalTensor()
    print(B)
    print(A-B)
    B.move_index_to_bottom(A.labels[0])
    print(A-B)
    A -= B
    print(A)

def test0063():
    A = random_tensor((2,3,4),["a","b","c"])
    print(A)
    U, S, V = tensor_svd(A,["a","b"])
    print(U,S,V)
    B = S * V * V.conj() * S.conj()
    print(B)
    S.data = S.data*S.data
    C = S.trace(S.labels[0],S.labels[1])
    print(B==C)
    b = B.to_scalar()
    print(b)
    B = scalar_to_tensor(b)
    print(B==C)


def test0064():
    A = random_tensor((3,4),["a","b"])
    Ah = A.adjoint("a")
    AhA = Ah["b"]*A["a"]
    trAhA = AhA.trace("a","b")
    print(A,Ah,AhA,trAhA)
    U, S, V = tensor_svd(A,["a"],svd_labels=["l","r"])
    Sh = S.adjoint("l")
    ShS = Sh["r"]*S["l"]
    trShS = ShS.trace("l","r")
    print(S,Sh,ShS,trShS)
    print(trShS==trAhA)


def test0065():
    A = random_tensor((3,4),["a","b"],dtype=complex)
    Ah = A.adjoint("a")
    AhA = Ah["b"]*A["a"]
    trAhA = AhA.trace("a","b")
    print(A,Ah,AhA,trAhA)
    U, S, V = tensor_svd(A,["a"],svd_labels=["l","l"])
    Sh = S.adjoint("l")
    ShS = Sh*S
    trShS = ShS.trace()
    print(S,Sh,ShS,trShS)
    print(trShS==trAhA)

def test0065():
    A = random_tensor((3,4),["a","b"],dtype=complex)
    print(A)
    U,S,V = tensor_svd(A,"a",svd_labels=[(1,0),(1,0)])
    print(U,S,V)
    B = U*S*V
    print(B)
    print(A==B)

def test0066():
    A = np.arange(12).reshape(3,4)
    print(A)
    B = A[xp.eye(*A.shape)==0]
    print(B)
    C = xp.linalg.norm(B)
    print(C)

test0066()