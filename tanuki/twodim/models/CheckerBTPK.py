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

        dataStr = f"Ptn2DCheckerTPK(\n" + dataStr + f")"

        return dataStr


    def renormalize_alg_LN(self, chi=10, normalize=True):
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

        if normalize:
            A2norm = A2.norm()
            A5norm = A5.norm()
            B2norm = B2.norm()
            B5norm = B5.norm()
            A2 /= A2norm
            A5 /= A5norm
            B2 /= B2norm
            B5 /= B5norm
            weight = A2norm*A5norm*B2norm*B5norm

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
        if normalize:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1), PowPowFloat(weight, 2, self.scale-1)
        else:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)



    def renormalize_alg_YGW1(self, chi=10, normalize=True, loop_compress_algname="YGW"):
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
        u1,u2 = U.sqrt2(B6)
        d1,d2 = D.sqrt2(A6)
        l1,l2 = L.sqrt2(B3)
        r1,r2 = R.sqrt2(A4)
        """
                u2   d2
                A1 R B1
               A2     B2
           l2 A3       B3 l1
              D         U
           r2 B4       A4 r1
               B5     A5  
                B6 L A6    
                u1   d1
        """
        # O(chi^7)
        E = B1*R*A1*u2*d2
        F = A3*D*B4*l2*r2
        G = B6*L*A6*u1*d1
        H = B3*U*A4*l1*r1
        """
                  ||
                   E
                A2   B2   
           == F         H ==
                B5   A5   
                   G
                   ||
        """
        # O(chi^8 * repeat)
        CBTPS = onedim.Cyc1DBTPS([E,F,G,H],[B2,A2,B5,A5])
        if normalize:
            weight = CBTPS.compress(chi=chi, transfer_normalize=True, algname=loop_compress_algname)
        else:
            CBTPS.compress(chi=chi, transfer_normalize=False, algname=loop_compress_algname)
        B2,A2,B5,A5 = tuple(CBTPS.bdts)
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
        if normalize:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1), PowPowFloat(weight, 2, self.scale-1)
        else:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)



    def renormalize_alg_YGW2(self, chi=10, normalize=True, loop_compress_algname="YGW"):
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
        u1,u2 = U.sqrt2(B6)
        d1,d2 = D.sqrt2(A6)
        l1,l2 = L.sqrt2(B3)
        r1,r2 = R.sqrt2(A4)
        """
                u2   d2
                A1 R B1
               A2     B2
           l2 A3       B3 l1
              D         U
           r2 B4       A4 r1
               B5     A5  
                B6 L A6    
                u1   d1
        """
        # O(chi^4)
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
        # O((chi^5 + chi^6) * repeat)
        CBTPS = onedim.Cyc1DBTPS([A1,A3,B4,B6,A6,A4,B3,B1],[R,A2,D,B5,L,A5,U,B2])
        if normalize:
            weight = CBTPS.compress(chi=chi, transfer_normalize=True, algname=loop_compress_algname)
        else:
            CBTPS.compress(chi=chi, transfer_normalize=False, algname=loop_compress_algname)
        R,A2,D,B5,L,A5,U,B2 = tuple(CBTPS.bdts)
        A1,A3,B4,B6,A6,A4,B3,B1 = tuple(CBTPS.tensors)
        
        # O(chi^6)
        X = (A1*R*B1)*(B6*L*A6)
        Y = (A3*D*B4)*(B3*U*A4)
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
        if normalize:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1), PowPowFloat(weight, 2, self.scale-1)
        else:
            return Ptn2DCheckerBTPK(X, Y, A2, A5, B5, B2, scale=self.scale-1)




    def renormalize(self, chi=10, normalize=True, algname="LN", loop_compress_algname="YGW"):
        if algname in ["LN", "Naive", "TRG"]:
            return self.renormalize_alg_LN(chi=chi, normalize=normalize)
        elif algname in ["YGW1"]:
            return self.renormalize_alg_YGW1(chi=chi, normalize=normalize, loop_compress_algname=loop_compress_algname)
        elif algname in ["YGW2","YGW"]:
            return self.renormalize_alg_YGW2(chi=chi, normalize=normalize, loop_compress_algname=loop_compress_algname)








    def calculate(self, chi=10,  normalize=True, algname="LN", loop_compress_algname="YGW"):
        temp = self
        if normalize:
            weight = PowPowFloat([])
        else:
            weight = 1.0
        while temp.scale > 0:
            if normalize:
                n,w = temp.renormalize(chi=chi, normalize=normalize, algname=algname, loop_compress_algname=loop_compress_algname)
                temp = n
                weight *= w
            else:
                temp = temp.renormalize(chi=chi, normalize=normalize, algname=algname, loop_compress_algname=loop_compress_algname)

        """
        [scale=1]
              U   D
            L A R B L
              D   U
        """
        weight *= (temp.A*temp.L*temp.R*temp.U*temp.D*temp.B).to_scalar()
        return weight









