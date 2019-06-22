import sys
sys.path.append('../../')
from tanuki import *
from tanuki.matrices import *
import numpy as np

def test0050():
    A = random_tensor((1,2,3,4,5),labels=["a","a","b","b","c"])
    print(A)
    A.move_index_to_bottom("a")
    print(A)
    A.move_indices_to_bottom(["b","b"])
    print(A)


def test0051():
    A = random_tensor((2,2,3,3,4),["a","a","b","b","c"])
    B = random_tensor((2,5),["a","d"])
    print(A*B)
    C = random_tensor((3,2,3),["b","a","b"])
    D = A*C
    print(D)
    E = D.trace()
    print(E)


def test0052():
    A = np.arange(16).reshape(2,2,2,2)
    B = np.arange(4).reshape(2,2)
    print(A)
    print(B)
    print(A*B)
    print(np.sum(A*B))

test0052()