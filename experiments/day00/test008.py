import sys
sys.path.append('../../')
from tanuki import *
from tanuki.matrices import *
from tanuki.lattices import *
import numpy as np


def test0080():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])
    s1 = S.to_tensor()
    S.canonize_end(normalize=False)
    s2 = S.to_tensor()
    print(s1)
    print(s2)
    print(s1==s2)


def test0081():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])
    s1 = S.to_tensor()
    S.canonize(normalize=False)
    s2 = S.to_tensor()
    print(s1==s2)


def test0082():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])

    S.canonize_end(normalize=False)
    V_L = S.get_left_eigenvector()
    print(V_L.is_prop_identity())


def test0083():
    chi=None
    relative_threshold=1e-14

    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    self = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])




    label_base = "TFL" #unique_label()
    inbra = label_base + "_inbra"
    inket = label_base + "_inket"
    outbra = label_base + "_outbra"
    outket = label_base + "_outket"
    dim = self.get_right_dim_site(len(self)-1)
    shape = self.get_right_shape_site(len(self)-1)
    rawl = self.get_right_labels_site(len(self)-1)

    TF_L = tni.identity_tensor(dim, shape, labels=[inket]+rawl)
    TF_L *= tni.identity_tensor(dim, shape, labels=[inbra]+aster_labels(rawl))
    for i in range(len(self)):
        TF_L *= self.bdts[i]
        TF_L *= self.bdts[i].adjoint(self.get_left_labels_bond(i),self.get_right_labels_bond(i), style="aster")
        TF_L *= self.tensors[i]
        TF_L *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")
    TF_L *= tni.identity_tensor(dim, shape, labels=[outket]+rawl)
    TF_L *= tni.identity_tensor(dim, shape, labels=[outbra]+aster_labels(rawl))
    #print("TF_L: ", TF_L.trace(inbra, inket))

    w_L, V_L = tnd.tensor_eigsh(TF_L, [outket,outbra], [inket,inbra])
    V_L.hermite(inket, inbra, assume_definite_and_if_negative_then_make_positive=True, inplace=True)
    #print("w*V:", w*V)
    #print("TF_L*V:", TF_L*V)

    V_L.split_index(inket, shape, rawl)
    V_L.split_index(inbra, shape, aster_labels(rawl))




    label_base = "TFR" #unique_label()
    inbra = label_base + "_inbra"
    inket = label_base + "_inket"
    outbra = label_base + "_outbra"
    outket = label_base + "_outket"
    dim = self.get_left_dim_site(0)
    shape = self.get_left_shape_site(0)
    rawl = self.get_left_labels_site(0)

    TF_R = tni.identity_tensor(dim, shape, labels=[inket]+rawl)
    TF_R *= tni.identity_tensor(dim, shape, labels=[inbra]+aster_labels(rawl))
    for i in range(len(self)-1, -1, -1):
        TF_R *= self.bdts[i+1]
        TF_R *= self.bdts[i+1].adjoint(self.get_left_labels_bond(i+1),self.get_right_labels_bond(i+1), style="aster")
        TF_R *= self.tensors[i]
        TF_R *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")
    TF_R *= tni.identity_tensor(dim, shape, labels=[outket]+rawl)
    TF_R *= tni.identity_tensor(dim, shape, labels=[outbra]+aster_labels(rawl))

    w_R, V_R = tnd.tensor_eigsh(TF_R, [outket,outbra], [inket,inbra])
    V_R.hermite(inbra, inket, assume_definite_and_if_negative_then_make_positive=True, inplace=True)
    V_R.split_index(inket, shape, rawl)
    V_R.split_index(inbra, shape, aster_labels(rawl))





    label_base = "TFM" #unique_label()
    inbra = label_base + "_inbra"
    inket = label_base + "_inket"
    outbra = label_base + "_outbra"
    outket = label_base + "_outket"
    dim = self.get_left_dim_site(0)
    shape = self.get_left_shape_site(0)
    rawl = self.get_left_labels_site(0)
    dim2 = self.get_right_dim_site(-1)
    shape2 = self.get_right_shape_site(-1)
    rawl2 = self.get_right_labels_site(-1)

    TF_M = tni.identity_tensor(dim, shape, labels=[inket]+rawl)
    TF_M *= tni.identity_tensor(dim, shape, labels=[inbra]+aster_labels(rawl))
    for i in range(len(self)):
        if i!=0:
            TF_M *= self.bdts[i]
            TF_M *= self.bdts[i].adjoint(self.get_left_labels_bond(i),self.get_right_labels_bond(i), style="aster")
        TF_M *= self.tensors[i]
        TF_M *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")
    TF_M *= tni.identity_tensor(dim2, shape2, labels=[outket]+rawl2)
    TF_M *= tni.identity_tensor(dim2, shape2, labels=[outbra]+aster_labels(rawl2))

    w_M, V_M = tnd.tensor_eigsh(TF_M, [outket, outbra], [inket,inbra])
    V_M.hermite(inket, inbra, assume_definite_and_if_negative_then_make_positive=True, inplace=True)
    V_M.split_index(inket, shape, rawl)
    V_M.split_index(inbra, shape, aster_labels(rawl))





    dl_label = "d_L"
    dr_label = "d_R"
    #print("V_L:", V_L)
    #print("V_R:", V_R)
    Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_right_labels_site(len(self)-1), aster_labels(self.get_right_labels_site(len(self)-1)), eigh_labels=dl_label)
    #print("sahen", tensou*Yh*d_L*Y)
    #print("uhen", Yh*d_L*Y)
    Y.unaster_labels(aster_labels(self.get_right_labels_site(len(self)-1)))
    #print(Yh*Y)
    X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_left_labels_site(0), aster_labels(self.get_left_labels_site(0)), eigh_labels=dr_label)
    Xh.unaster_labels(aster_labels(self.get_left_labels_site(0)))
    #print(Xh*X)
    l0 = self.bdts[0]
    l0h = l0.adjoint(self.get_left_labels_bond(i+1),self.get_right_labels_bond(i+1), style="aster")
    G = d_L.sqrt() * Yh * l0 * X * d_R.sqrt()
    Gh = d_L.sqrt() * Y * l0 * Xh * d_R.sqrt()

    """
    print(V_L)
    print(l0)
    print(l0h)
    """
    """
    print(TF_M)
    t = TF_M
    t = l0["v0"] * t[inbra]
    t = l0h["v0*"] * t[inket]
    print(t)
    print(TF_L)
    """
    #print(V_L)
    #print(TF_L)
    """
    s = V_L[["v0", "v0*"]] * TF_L[["TFL_inket", "TFL_inbra"]]
    print("w_L*V_L", w_L*V_L)
    print("V_L*TF_L", s)

    
    A = d_L.sqrt()*Yh*l0
    Ah = d_L.sqrt()*Y*l0
    t = A["v0"]*TF_M[inket]
    t = Ah["v0"]*t[inbra]
    t.trace(inplace=True)
    t = t[outket] * (Y*d_L.sqrt().inv())["v0"] 
    t = t[outbra] * (Yh*d_L.sqrt().inv())["v0"] 
    print(t)
    
    u = d_L.sqrt().inv() * Yh * (Y * d_L.sqrt())
    print(u)
    """
    
    A = G*d_R.sqrt().inv()*Xh
    Ah = Gh*d_R.sqrt().inv()*X
    t = A["v0"]*TF_M[inket]
    t = Ah["v0"]*t[inbra]
    t.trace(inplace=True)
    t = t[outket] * (Y*d_L.sqrt().inv())["v0"] 
    t = t[outbra] * (Yh*d_L.sqrt().inv())["v0"] 
    print(t)
    

    U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi, relative_threshold=relative_threshold)
    M = Y * d_L.inv().sqrt() * U
    N = V * d_R.inv().sqrt() * Xh
    # l0 == M*S*N

    """
    A = TF_M
    A = A * Xh * d_R.inv().sqrt() * G * Y * d_L.inv().sqrt()
    A = A * X * d_R.inv().sqrt() * G.
    """


    self.bdts[0] = S
    self.tensors[0] = N * self.tensors[0]
    self.tensors[len(self)-1] = self.tensors[len(self)-1] * M




def test0084():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])

    S.canonize(normalize=False)
    print(S.is_canonical())


def test0085():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])

    S.canonize_end()
    w_L, V_L = S.get_left_eigen()
    print(w_L)
    print(V_L)
    assert abs(w_L-1)<1e-10
    assert V_L.is_prop_identity()

def test0086():
    l0 = random_diagonalTensor(2, ["v0", "v0"])
    t0 = random_tensor((2,3,2),["v0", "p0", "v1"], dtype=complex)
    l1 = random_diagonalTensor(2, ["v1", "v1"])
    t1 = random_tensor((2,3,2),["v1", "p1", "v2"], dtype=complex)
    l2 = random_diagonalTensor(2, ["v2", "v2"])
    t2 = random_tensor((2,3,2),["v2", "p2", "v0"], dtype=complex)
    S = Inf1DBTPS([t0,t1,t2],[l0,l1,l2])

    S.canonize()
    w_L, V_L = S.get_left_eigen()
    print(w_L)
    print(V_L)
    assert abs(w_L-1)<1e-10
    assert V_L.is_prop_identity()

test0085()