import sys
sys.path.append('../../')
from tanuki import *
import numpy as np

def test0030():
    print(unique_label())
    print(unique_label())
    print(unique_label())


def test0031():
    print(Tensor(np.arange(10).reshape(2,5)))


def test0032():
    A = Tensor(np.arange(64).reshape(8,8))
    print(A)
    U,S,V = truncated_svd(A,A.labels[0],chi=2)
    print(U,S,V)
    B = U*S*V
    print(B)
    print(A==B)


def test0033():
    A = Tensor(np.arange(30).reshape(5,6))
    print(A)
    L,Q = tensor_lq(A,A.labels[0])
    print(L,Q)
    B = L*Q
    print(B)
    print(A==B)


def test0034():
    A = Tensor(np.arange(30).reshape(5,6))
    print(A)
    L,Q = tensor_lq(A,A.labels[0],lq_labels=["lq_l_right","lq_q_left"])
    print(L,Q)
    B = L["lq_l_right"]*Q["lq_q_left"]
    print(B)
    print(A==B)



def test0035():
    A = Tensor([[2,9,4],[7,5,3],[6,1,8]])
    print(A)

    U,S,V = truncated_svd(A,A.labels[0],chi=3)
    print(U,S,V)
    B = U*S*V
    print(B)
    print(A==B)

    U,S,V = truncated_svd(A,A.labels[0],chi=2)
    print(U,S,V)
    B = U*S*V
    print(B)
    print(A==B)

def test0036():
    A = Tensor([[1,2],[3,4]],["a","b"])
    print(A)
    A.replace_labels(["a","b"],["b","a"])
    print(A)
    A.replace_labels(["a","b"],["b","a"])
    print(A)

def test0037():
    A = Tensor([[1+1j,2+4j],[3-4j,5]],["a","b"])
    print(A)
    B = A.hermite("a")
    print(B)
    C = A.antihermite("b")
    print(C)
    print(B+C)
    print(B+C==A)

def test0038():
    A = Tensor([[[1+1j,2+4j],[3-4j,5]],[[0,9],[3,1j]]],["a","b","c"])
    print(A)
    B = A.hermite("a",["b"])
    print(B)
    C = A.antihermite(["a"],["b"])
    print(C)
    print(B+C)
    print(B+C==A)

test0038()