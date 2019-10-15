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





class Ptr2DCheckerBTPK:
    """
    [scale=2]
          U   D
        L A R B L
          D   U
        R B L A R
          U   D
    """
    def __init__(self, A, B, L, R, U, D, scale=5):
        self.A = A
        self.B = B
        self.L = L
        self.R = R
        self.U = U
        self.D = D
        self.scale = scale

    def __str__(self):
        dataStr = ""
        dataStr += "A = " + str(self.A) + "\n"
        dataStr += "B = " + str(self.B) + "\n"
        dataStr += "L = " + str(self.L) + "\n"
        dataStr += "R = " + str(self.R) + "\n"
        dataStr += "U = " + str(self.U) + "\n"
        dataStr += "D = " + str(self.D) + "\n"
        dataStr += "scale=" + str(self.scale) + "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Ptr2DCheckerTPK(\n" + dataStr + f")"

        return dataStr


    def renormalize_alg_LN(self, chi=10):
        """
        Tensor renormalization group approach to 2D classical lattice models
        Michael Levin, Cody P. Nave
        https://arxiv.org/abs/cond-mat/0611687
        """
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D

        # O(chi^5)
        A1,A2,A3 = A.svd(intersection_list(A.labels, R.labels+U.labels), chi=chi, svd_labels=2)
        A4,A5,A6 = A.svd(intersection_list(A.labels, R.labels+U.labels), chi=chi, svd_labels=2)
        B1,B2,B3 = B.svd(intersection_list(B.labels, R.labels+D.labels), chi=chi, svd_labels=2)
        B4,B5,B6 = B.svd(intersection_list(B.labels, R.labels+D.labels), chi=chi, svd_labels=2)

        """
                 U   D
                A1 R B1
               A2     B2
            L A3       B3 L
              D         U
            R B4       A4 R
               B5     A5  
                B6 L A6    
                 U   D
        """
        # O(chi^6)
        X = (B6*L*A6)*U*D*(B1*R*A1)
        Y = (B3*U*A4)*L*R*(A3*D*B4)
        """
                  X
               A2   B2
              Y       Y 
               B5   A5
                  X
        """
        """
            X A5 Y
            B2   B5
            Y A2 X
        """
        return Ptr2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)



    def renormalize_alg_YGW(self, chi=10):
        """
        Loop optimization for tensor network renormalization
        Shuo Yang, Zheng-Cheng Gu, Xiao-Gang Wen
        https://arxiv.org/abs/1512.04938
        """
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D

        # O(chi^6)
        A1,A2,A3 = A.svd(intersection_list(A.labels, R.labels+U.labels), svd_labels=2)
        A4,A5,A6 = A.svd(intersection_list(A.labels, R.labels+U.labels), svd_labels=2)
        B1,B2,B3 = B.svd(intersection_list(B.labels, R.labels+D.labels), svd_labels=2)
        B4,B5,B6 = B.svd(intersection_list(B.labels, R.labels+D.labels), svd_labels=2)

        """
                 U   D
                A1 R B1
               A2     B2
            L A3       B3 L
              D         U
            R B4       A4 R
               B5     A5  
                B6 L A6    
                 U   D
        """
        # O(chi^7)
        E = B1*R*A1*U
        F = B3*U*A4*L
        G = B6*L*A6*D
        H = A3*D*B4*R
        """
                  ||
                   E
                A2   B2   
           == H         F ==
                B5   A5   
                   G
                   ||
        """
        # O(chi^12) #OMG!!
        CBTPS = onedim.Cyc1DBTPS([E,F,G,H],[A2,B2,A5,B5])
        CBTPS.universally_canonize(chi=chi, transfer_normalize=False)
        A2,B2,A5,B5 = tuple(CBTPS.bdts)
        E,F,G,H = tuple(CBTPS.tensors)
        
        X = E*G
        Y = F*H
        """
                  X
               A2   B2
              Y       Y 
               B5   A5
                  X
        """
        """
            X A5 Y
            B2   B5
            Y A2 X
        """
        return Ptr2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)



    def renormalize(self, chi=10, algname="LN"):
        if algname in ["LN", "Naive", "TRG"]:
            return self.renormalize_alg_LN(chi=chi)
        elif algname in ["YGW", "Loop-TNR"]:
            return self.renormalize_alg_YGW(chi=chi)








    def calculate(self, chi=10, algname="LN"):
        re = self
        while re.scale > 0:
            re = re.renormalize(chi=chi, algname=algname)

        """
        [scale=1]
              U   D
            L A R B L
              D   U
        """
        return re.A*re.L*re.R*re.U*re.D*re.B









