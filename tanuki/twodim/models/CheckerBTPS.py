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



    def super_orthogonalize(self, conv_rtol=1e-11, conv_atol=1e-14, maxiter=100, normalize_where="keep_overall", memo=None):
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
            """
            if lastA is not None and A.__eq__(lastA, skipLabelSort=True):
                break
            lastA = A
            """
            if lastL is not None and L.__eq__(lastL, skipLabelSort=True, check_rtol=conv_rtol, check_atol=conv_atol):
                break
            lastL = L
        memo["iter_times"] = iteri+1

        cbl = (A*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cbr = (A*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cbu = (A*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cbd = (A*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cal = (B*R*U*D).is_prop_right_semi_unitary(rows=L, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        car = (B*L*U*D).is_prop_right_semi_unitary(rows=R, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cau = (B*L*R*D).is_prop_right_semi_unitary(rows=U, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)
        cad = (B*L*R*U).is_prop_right_semi_unitary(rows=D, check_rtol=conv_rtol*1e3, check_atol=conv_atol*1e3)["factor"]**(0.5)


        cccaaa__b = cal*car*cau*cad
        cccbbb__a = cbl*cbr*cbu*cbd
        # a__b = (cccaaa__b/cccbbb__a)**0.25
        # print(a__b) # kore daitai 1 ni narukedo tabun genmitsu na 1 deha naiyone?

        if normalize_where in ["keep_overall","A","B"]:
            if normalize_where == "keep_overall": # assert keeps A*L*R*U*D*B
                c = 1.0/weight
                a = ((cccaaa__b**3 * cccbbb__a)**0.25 / c**3)**0.5
                b = ((cccbbb__a**3 * cccaaa__b)**0.25 / c**3)**0.5
            elif normalize_where == "A": # assert A.norm() == 1
                a = 1.0/A.norm()
                b = (cccbbb__a / cccaaa__b )**0.25 * a
                c = (cccaaa__b * b)**(1/3) / a
            elif normalize_where == "B":
                b = 1.0/B.norm()
                a = (cccaaa__b / cccbbb__a )**0.25 * b
                c = (cccbbb__a * a)**(1/3) / b
            ca = c*a
            cb = c*b

        if normalize_where in ["L","R","U","D"]:
            if normalize_where == "L":
                l = 1.0/L.norm()
                ca = cal/l
                cb = cbl/l
            elif normalize_where == "R":
                r = 1.0/R.norm()
                ca = car/r
                cb = cbr/r
            elif normalize_where == "U":
                u = 1.0/U.norm()
                ca = cau/u
                cb = cbu/u
            elif normalize_where == "D":
                d = 1.0/D.norm()
                ca = cad/d
                cb = cbd/d
            c = cccbbb__a * ca / cb**3
            a = ca / c
            b = cb / c

        if normalize_where != "L": l = cbl / cb
        if normalize_where != "R": r = cbr / cb
        if normalize_where != "U": u = cbu / cb
        if normalize_where != "D": d = cbd / cb

        A *= a
        B *= b
        L *= l
        R *= r
        U *= u
        D *= d
        self.A,self.B,self.L,self.R,self.U,self.D = A,B,L,R,U,D

        print(memo)

        return 1.0