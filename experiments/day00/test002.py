import sys
sys.path.append('../../')
from tanuki import *
import numpy as np

def test0020():
    a = np.arange(6) + np.arange(6)*1j
    a = Tensor(a,["a"])
    a.split_index("a",[2,3],["a","b"])
    print(a)
    print(a.conjugate())
    print(a)
    a.conjugate(inplace=True)
    print(a)


def test0021():
    a = np.arange(6) + np.arange(6)*1j
    a = Tensor(a,["a"])
    a.split_index("a",[2,3],["a","b"])

    b = np.ones(6)
    b = Tensor(b, ["a"])
    b.split_index("a",[3,2],["b","a"])

    print(a)
    print(b)
    print(a+b)
    print(a)
    print(b)

def test0022():
    a = np.arange(6) + np.arange(6)*1j
    a = Tensor(a,["a"])
    a.split_index("a",[2,3],["a","b"])

    try:
        b = np.ones(6)
        b = Tensor(b, ["a"])
        b.split_index("a",[6,1],["b","a"])
        print(a+b)
    except Exception as e:
        print(e)

    try:
        b = np.ones(6)
        b = Tensor(b, ["a"])
        b.split_index("a",[2,3],["d","a"])
        print(a+b)
    except Exception as e:
        print(e)

    b = np.ones(6)
    b = Tensor(b, ["a"])
    b.split_index("a",[2,3],["d","a"])
    print(a.__add__(b,skipLabelSort=True))


def test0023():
    A = Tensor(np.arange(6).reshape(2,3), ["l","r"])
    B = Tensor(np.arange(6).reshape(3,2), ["l","r"])
    print(A,B)
    print(contract(A,B,"r","l"))
    print(contract(A,B,["r"],["l"]))
    print(A["r"]*B["l"])
    print(A[["r"]]*B[["l"]])
    print(A.contract(B,"r","l"))
    print(A,B)
    A.contract(B,"r","l",inplace=True)
    print(A,B)


def test0024():
    A = Tensor(np.arange(20).reshape(2,2,5), [1,2,3])
    print(A)
    print(A.trace(1,2))
    print(A)
    A.tr(1,2,inplace=True)
    print(A)

def test0025():
    mat = np.arange(12).reshape(4,3)
    print(mat)
    ten = matrix_to_tensor(mat, (2,2,3), ["a","b","c"])
    ten.move_index_to_bottom("b")
    mat2 = tensor_to_matrix(ten, ["a","b"],["c"])
    print(mat2)

def test0026():
    vec = np.arange(12)
    print(vec)
    ten = vector_to_tensor(vec, (2,2,3), ["a","b","c"])
    ten.move_index_to_bottom("b")
    vec2 = tensor_to_vector(ten, ["a","b","c"])
    print(vec2)

def test0027():
    A = Tensor(np.arange(64).reshape(8,8),["al","ar"])
    print(A)
    U,S,V = tensor_svd(A,"al","ar")
    print(U,S,V)
    A = (U["svd_ur"]*S["svd_sl"])["svd_sr"]*V["svd_vl"]
    print(A)

def test0028():
    A = Tensor(np.arange(64).reshape(8,8),["al","ar"])
    print(A)
    U,S,V = truncated_svd(A,"al","ar",chi=2)
    print(U,S,V)
    A = (U["svd_ur"]*S["svd_sl"])["svd_sr"]*V["svd_vl"]
    print(A)

def test0029():
    A = Tensor(np.arange(64).reshape(8,8),["al","ar"])
    print(A)
    Q,R = tensor_qr(A,"al","ar")
    print(Q,R)
    A = Q["qr_qr"]*R["qr_rl"]
    print(A)


def test0030():
    A = Tensor([[0,1,2,3],[1,2,4,6],[2,4,8,12]],["al","ar"])
    print(A)
    L,Q = tensor_lq(A,"al","ar")
    print(L,Q)
    A = L["lq_lr"]*Q["lq_ql"]
    print(A)


def test0031():
    A = Tensor(np.arange(18).reshape(2,3,1,3,1,1),["a","b","c","d","e","f"])
    print(A)
    A.remove_all_dummy_indices()
    print(A)
    A.add_dummy_index("c")
    A.remove_all_dummy_indices(["d"])
    print(A)
    
test0031()