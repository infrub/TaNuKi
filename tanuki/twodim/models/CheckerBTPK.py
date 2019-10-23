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





class Ptn2DCheckerBTPK:
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

    def __str__(self, nodata=False):
        dataStr = ""
        dataStr += "A = " + self.A.__str__(nodata=nodata) + "\n"
        dataStr += "B = " + self.B.__str__(nodata=nodata) + "\n"
        dataStr += "L = " + self.L.__str__(nodata=nodata) + "\n"
        dataStr += "R = " + self.R.__str__(nodata=nodata) + "\n"
        dataStr += "U = " + self.U.__str__(nodata=nodata) + "\n"
        dataStr += "D = " + self.D.__str__(nodata=nodata) + "\n"
        dataStr += "scale=" + str(self.scale) + "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Ptn2DCheckerTPK(\n" + dataStr + f")"

        return dataStr



    def renormalize(self, chi=10, normalize=True, env_choice="half", contract_medium=False, loop_truncation_algname="canonize", drill_parity=0):
        """
        Loop optimization for tensor network renormalization
        Shuo Yang, Zheng-Cheng Gu, Xiao-Gang Wen
        https://arxiv.org/abs/1512.04938
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

        if env_choice=="no":
            pass
        elif env_choice=="half":
            u1,u2 = U.sqrt2(B6)
            d1,d2 = D.sqrt2(A6)
            l1,l2 = L.sqrt2(B3)
            r1,r2 = R.sqrt2(A4)
            A1 *= u2
            A3 *= l2
            B4 *= r2
            B6 *= u1
            A6 *= d1
            A4 *= r1
            B3 *= l1
            B1 *= d2
            """
                     |   |
                    A1 R B1
                   A2     B2
               -- A3       B3 --
                  D         U
               -- B4       A4 --
                   B5     A5  
                    B6 L A6    
                     |   |
            """
        else:
            raise ArgumentError


        if contract_medium:
            # O(chi^7)
            E = A1*R*B1
            F = A3*D*B4
            G = B6*L*A6
            H = B3*U*A4

            # O(chi^8 * repeat)
            CBTPS = onedim.Cyc1DBTPS([E,F,G,H],[B2,A2,B5,A5])
            if normalize:
                weight = CBTPS.truncate(chi=chi, normalize=True, algname=loop_truncation_algname)
            else:
                CBTPS.truncate(chi=chi, normalize=False, algname=loop_truncation_algname)

            B2,A2,B5,A5 = tuple(CBTPS.bdts)
            E,F,G,H = tuple(CBTPS.tensors)

        else:
            # O((chi^5 + chi^6) * repeat)
            CBTPS = onedim.Cyc1DBTPS([A1,A3,B4,B6,A6,A4,B3,B1],[R,A2,D,B5,L,A5,U,B2])
            if normalize:
                weight = CBTPS.truncate(chi=chi, normalize=True, algname=loop_truncation_algname)
            else:
                CBTPS.truncate(chi=chi, normalize=False, algname=loop_truncation_algname)
            R_,A2,D_,B5,L_,A5,U_,B2 = tuple(CBTPS.bdts)
            A1,A3,B4,B6,A6,A4,B3,B1 = tuple(CBTPS.tensors)

            # O(chi^6)
            E = A1*R_*B1
            F = A3*D_*B4
            G = B6*L_*A6
            H = B3*U_*A4

        """
                  ||
                   E
                A2   B2   
           == F         H ==
                B5   A5   
                   G
                   ||
        """

        # O(chi^6)
        if env_choice=="no":
            X = E*U*D*G
            Y = F*L*R*H
        elif env_choice=="half":
            X = E*G
            Y = F*H
        """
                  X
               A2   B2
              Y       Y 
               B5   A5
                  X
        """

        if drill_parity % 2 == 0:
            A,B,L,R,U,D = X,Y,B5,B2,A5,A2
            """
                X B2 Y
                A2   A5
                Y B5 X
            """
        else:
            A,B,L,R,U,D = X,Y,A2,A5,B5,B2
            """
                X A5 Y
                B2   B5
                Y A2 X
            """


        if normalize:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1), PowPowFloat(weight, 2, self.scale-1)
        else:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)




    def calculate(self, chi=10,  normalize=True, **kwargs):
        temp = self
        if normalize:
            weight = PowPowFloat([])
        else:
            weight = 1.0
        while temp.scale > 0:
            if normalize:
                n,w = temp.renormalize(chi=chi, normalize=normalize, **kwargs)
                temp = n
                weight *= w
            else:
                temp = temp.renormalize(chi=chi, normalize=normalize, **kwargs)

        """
        [scale=1]
              U   D
            L A R B L
              D   U
        """
        weight *= (temp.A*temp.L*temp.R*temp.U*temp.D*temp.B).to_scalar()
        return weight









