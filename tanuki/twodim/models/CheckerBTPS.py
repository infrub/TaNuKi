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



    def super_orthogonalize_ver1(self, maxiter=100):
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D

        for iteri in range(maxiter):
            A,L,B = tnd.tensor_svd_again_in_bdts(A,L,B,[R,U,D],[R,U,D])
            print(A,B)
            A,R,B = tnd.tensor_svd_again_in_bdts(A,R,B,[L,U,D],[L,U,D])
            print(A,B)
            A,U,B = tnd.tensor_svd_again_in_bdts(A,U,B,[L,R,D],[L,R,D])
            print(A,B)
            A,D,B = tnd.tensor_svd_again_in_bdts(A,D,B,[L,R,U],[L,R,U])
            print(A,B)

        self.A,self.B,self.L,self.R,self.U,self.D = A,B,L,R,U,D



    def super_orthogonalize_ver2(self):
        """
        "i" is "i" of "inverse"

                        U                 D
                       TD                SU
                       TDi               SUi
        SLi SL L TR TRi A TLi TL R SR SRi B SLi SL L
                       TUi               SDi
                       TU                SD
                        D                 U
                       SD                TU
                       SDi               TUi
        TLi TL R SR SRi B SLi SL L TR TRi A TLi TL R
                       SUi               TDi
                       SU                TD
                        U                 D
        """
        """
        tshape = self.get_right_shape_site(bde-1)
        tlabels = self.get_ket_right_labels_site(bde-1)
        trdim = soujou(tshape)
        trlabel = unique_label() if edge_label is None else edge_label
        T = tni.random_tensor((trdim,)+tshape, [trlabel]+tlabels)"""

    super_orthogonalize = super_orthogonalize_ver1

