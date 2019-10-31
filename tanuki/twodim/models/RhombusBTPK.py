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





class Ptn2DRhombusBTPK:
    """
    Similar to Checker.

    Contains 2**(width_scale+height_scale) of A, 2**(width_scale+height_scale) of B, so 2**(width_scale+height_scale+1) of A or B.

    [width_scale=1, height_scale=1]
       L   U   L   U
         A       A
       D   R   D   R
             B       B
       L   U   L   U
         A       A
       D   R   D   R
             B       B
    """
    def __init__(self, A, B, L, R, U, D, width_scale=5, height_scale=5):
        if width_scale<0 or height_scale<0: raise ValueError
        self.A = A
        self.B = B
        self.L = L
        self.R = R
        self.U = U
        self.D = D
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
        dataStr += "width_scale=" + str(self.width_scale) + "\n"
        dataStr += "height_scale=" + str(self.height_scale) + "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Ptn2DRhombusTPK(\n" + dataStr + f")"

        return dataStr



    def renormalize(self, chi=10, normalize=True, env_choice="half", contract_before_truncate=False, loop_truncation_algname="canonize", drill_parity=0):
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D

        # O(chi^5)
        A1,A2,A3 = A.svd(intersection_list(A.labels, R.labels+U.labels), chi=chi, svd_labels=2)
        A4,A5,A6 = A.svd(intersection_list(A.labels, R.labels+U.labels), chi=chi, svd_labels=2)
        B1,B2,B3 = B.svd(intersection_list(B.labels, R.labels+D.labels), chi=chi, svd_labels=2)
        B4,B5,B6 = B.svd(intersection_list(B.labels, R.labels+D.labels), chi=chi, svd_labels=2)

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
        else:
            raise ArgumentError


        if contract_before_truncate:
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

        # O(chi^6)
        if env_choice=="no":
            X = E*U*D*G
            Y = F*L*R*H
        elif env_choice=="half":
            X = E*G
            Y = F*H
        """
            Y A2 X
            B5  B2
            X A5 Y
        """
        if drill_parity % 2 == 0:
            A,B,L,R,U,D = Y,X,A5,A2,B2,B5
        else:
            A,B,L,R,U,D = X,Y,A2,A5,B5,B2


        from tanuki.twodim.models.CheckerBTPK import Ptn2DCheckerBTPK
        if normalize:
            return Ptn2DCheckerBTPK(A,B,L,R,U,D, width_scale=self.width_scale, height_scale=self.height_scale), ef_pow(weight, 2**(self.width_scale+self.height_scale-1))
        else:
            return Ptn2DCheckerBTPK(A,B,L,R,U,D, width_scale=self.width_scale, height_scale=self.height_scale)




    def calculate(self, chi=10,  normalize=True, **kwargs):
        #print("AL",self.A.norm(), self.L.norm())
        A,B,L,R,U,D = self.A,self.B,self.L,self.R,self.U,self.D

        if self.height_scale==0:
            """
            [height_scale=0]
               L   U   L   U   L   U   L   U
                 A       A       A       A
               D   R   D   R   D   R   D   R
                     B       B       B       B
            """
            sev_labels = intersection_list(B.labels, L.labels+D.labels)
            left_label = unique_label()
            right_label = unique_label()
            G = A*L*D
            G = G.fuse_indices(sev_labels, left_label)
            G = G*U*R*B
            G = G.fuse_indices(sev_labels, right_label)

            for _ in range(self.width_scale):
                G = G[right_label]*G[left_label]

            G = G.trace(left_label, right_label)

            return ExpFloat( G.to_scalar() )

        if self.width_scale==0:
            """
            [width_scale=0]
               L   U
                 A
               D   R
                     B
               L   U
                 A
               D   R
                     B
            """
            sev_labels = intersection_list(B.labels, L.labels+U.labels)
            up_label = unique_label()
            down_label = unique_label()
            G = A*L*U
            G = G.fuse_indices(sev_labels, up_label)
            G = G*D*R*B
            G = G.fuse_indices(sev_labels, down_label)

            for _ in range(self.height_scale):
                G = G[down_label]*G[up_label]

            G = G.trace(up_label, down_label)

            return ExpFloat( G.to_scalar() )


        if normalize:
            temp,w = self.renormalize(chi=chi, normalize=normalize, **kwargs)
            return temp.calculate(chi=chi, normalize=normalize, **kwargs) * w
        else:
            temp = self.renormalize(chi=chi, normalize=normalize, **kwargs)
            return temp.calculate(chi=chi, normalize=normalize, **kwargs)




