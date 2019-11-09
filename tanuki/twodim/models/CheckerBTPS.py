import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki import onedim
import textwrap
from math import sqrt
import logging
import numpy as np





class Ptn2DCheckerBTPS:
    """
    Contains 2**(width_scale+height_scale-1) of A, 2**(width_scale+height_scale-1) of B, so 2**(width_scale+height_scale) of A or B.

    [width_scale=1, height_scale=1]
          U   D
        L A R B L
          D   U
        R B L A R
          U   D
    """
    def __init__(self, A, B, L, R, U, D, width_scale=5, height_scale=5):
        if width_scale<1 or height_scale<1: raise ValueError
        self.A = A
        self.B = B
        self.L = L
        self.R = R
        self.U = U
        self.D = D
        self.A_phys_labels = diff_list(A.labels, L.labels+R.labels+U.labels+D.labels, assume_included=False)
        self.B_phys_labels = diff_list(B.labels, L.labels+R.labels+U.labels+D.labels, assume_included=False)
        self.width_scale = width_scale
        self.height_scale = height_scale

    def __str__(self, nodata=False):
        dataStr = ""
        dataStr += "A = " + self.A.__str__(nodata=nodata) + "\n"
        dataStr += "B = " + self.B.__str__(nodata=nodata) + "\n"
        dataStr += "L = " + self.L.__str__(nodata=nodata) + "\n"
        dataStr += "R = " + self.R.__str__(nodata=nodata) + "\n"
        dataStr += "U = " + self.U.__str__(nodata=nodata) + "\n"
        dataStr += "D = " + self.D.__str__(nodata=nodata) + "\n"
        dataStr += "A_phys_labels=" + str(self.A_phys_labels) + "\n"
        dataStr += "B_phys_labels=" + str(self.B_phys_labels) + "\n"
        dataStr += "width_scale=" + str(self.width_scale) + "\n"
        dataStr += "height_scale=" + str(self.height_scale) + "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Ptn2DCheckerTPS(\n" + dataStr + f")"

        return dataStr


    def locally_canonize(self, where, chi=None, normalize=True):
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D
        weight = 1.0
        if where=="L":
            A,L,B = tnd.tensor_svd_again_in_bdts(A,L,B,[R,U,D],[R,U,D], chi=chi)
            if normalize: weight *= L.normalize(inplace=True)
        elif where=="R":
            A,R,B = tnd.tensor_svd_again_in_bdts(A,R,B,[L,U,D],[L,U,D], chi=chi)
            if normalize: weight *= R.normalize(inplace=True)
        elif where=="U":
            A,U,B = tnd.tensor_svd_again_in_bdts(A,U,B,[L,R,D],[L,R,D], chi=chi)
            if normalize: weight *= U.normalize(inplace=True)
        elif where=="D":
            A,D,B = tnd.tensor_svd_again_in_bdts(A,D,B,[L,R,U],[L,R,U], chi=chi)
            if normalize: weight *= D.normalize(inplace=True)

        if normalize:
            a = (A*L*R*U*D).norm()
            b = (B*L*R*U*D).norm()
            A /= a
            B /= b
            weight *= a * b
        self.A,self.B,self.L,self.R,self.U,self.D = A,B,L,R,U,D
        return weight


    def locally_canonize_all(self, conv_rtol=1e-11, conv_atol=1e-14, maxiter=100, normalize=True, memo=None):
        if memo is None: memo={}
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D
        weight = 1.0
        #lastA = None
        lastL = None

        for iteri in range(maxiter):
            A,L,B = tnd.tensor_svd_again_in_bdts(A,L,B,[R,U,D],[R,U,D])
            weight *= A.normalize(inplace=True) * L.normalize(inplace=True) * B.normalize(inplace=True)
            A,R,B = tnd.tensor_svd_again_in_bdts(A,R,B,[L,U,D],[L,U,D])
            weight *= A.normalize(inplace=True) * R.normalize(inplace=True) * B.normalize(inplace=True)
            A,U,B = tnd.tensor_svd_again_in_bdts(A,U,B,[L,R,D],[L,R,D])
            weight *= A.normalize(inplace=True) * U.normalize(inplace=True) * B.normalize(inplace=True)
            A,D,B = tnd.tensor_svd_again_in_bdts(A,D,B,[L,R,U],[L,R,U])
            weight *= A.normalize(inplace=True) * D.normalize(inplace=True) * B.normalize(inplace=True)
            if lastL is not None and L.__eq__(lastL, skipLabelSort=True, check_rtol=conv_rtol, check_atol=conv_atol):
                break
            lastL = L
        memo["iter_times"] = iteri+1

        a = (A*L*R*U*D).norm()
        b = (B*L*R*U*D).norm()
        A /= a
        B /= b
        weight *= a
        weight *= b

        # assert (A*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"] ~= 1.0

        if not normalize:
            tensor_m = weight**1.5
            bdt_m = weight**(-0.5)
            A *= tensor_m
            B *= tensor_m
            L *= bdt_m
            R *= bdt_m
            U *= bdt_m
            D *= bdt_m
            weight = 1.0

        self.A,self.B,self.L,self.R,self.U,self.D = A,B,L,R,U,D

        print(memo)

        return weight

    super_orthogonalize = locally_canonize_all


    def apply(self, mpo, where, chi=None, normalize=True):
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D
        C = {"L":L,"R":R,"U":U,"D":D}[where]

        mpo = mpo.to_BTPO()

        A = A[self.A_phys_labels] * mpo.tensors[0][mpo.physin_labelss[0] : mpo.physout_labelss[0] : self.A_phys_labels]
        C = C * mpo.bdts[1]
        B = B[self.B_phys_labels] * mpo.tensors[1][mpo.physin_labelss[1] : mpo.physout_labelss[1] : self.B_phys_labels]

        if where=="L": L=C
        if where=="R": R=C
        if where=="U": U=C
        if where=="D": D=C
        self.A,self.B,self.L,self.R,self.U,self.D = A,B,L,R,U,D

        return self.locally_canonize(where,chi=chi,normalize=normalize)

