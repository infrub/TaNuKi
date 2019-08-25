import sys
sys.path.append('../../')
from tanuki import *
from tanuki.matrices import *
import numpy as np

def test0040():
    A = random_tensor((2,3,4))
    print(A)
    B = random_tensor_like(A)
    print(B)
    print(A+B)
    assert A+B==B+A

def test0041():
    A = random_tensor((2,3,4),dtype=complex)
    print(A)
    B = zeros_tensor_like(A)
    print(B)
    print(A+B)
    assert A+B==A

def test0042():
    A = random_tensor((2,3,4),base_label="a",dtype=complex)
    print(A)
    B = identity_tensor(3,"a_1catch","a_1")
    print(B)
    C = B["a_1catch"]*A["a_1"]
    print(C)
    print(A-C)
    assert A==C

def test0043():
    A = Tensor([[sigmax,sigmax],[sigmay,sigmaz]])
    print(A)
    A.data[0,0,0,0]=4
    print(A)


test0043()