import sys
sys.path.append('../../')
from tanuki import *
from tanuki.matrices import *
from tanuki.lattices import *
import numpy as np

def test0070():
    A = random_tensor((2,3,4),["a","b","c"])
    print(A)
    U, S, V = tensor_svd(A,["a","b"])
    S = S.to_tensor()
    print(U,S,V)
    print(U.is_left_unitary(["a","b"]))
    print(S.is_diagonal())
    print(V.is_right_unitary("c"))

def test0071():
    A = random_tensor((2,3),["p0","v0"])
    B = random_tensor((3,2,3),["v0","p1","v1"])
    C = random_tensor((3,2),["v1","p2"])
    S = Opn1DTPS([A,B,C])
    print(S)
    s = S.to_tensor()
    print(s)

    T = S.to_BTPS()
    print(T)
    T.both_canonize(end_dealing="no")
    print(T)
    t = T.to_tensor()
    print(t)
    assert s==t

    print(T)
    assert T.is_both_canonical(end_dealing="no")

def test0072():
    A = random_tensor((2,3),["p0","v0"])
    B = random_tensor((3,2,3),["v0","p1","v1"])
    C = random_tensor((3,2),["v1","p2"])
    S = Opn1DTPS([A,B,C])
    print(S)
    s = S.to_tensor()
    print(s)

    T = S.to_BTPS()
    print(T)
    T.both_canonize(end_dealing="normalize")
    print(T)
    assert T.is_both_canonical(end_dealing="normalize")


test0071()