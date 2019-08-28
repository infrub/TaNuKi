from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *
import warnings
import time
import math
import numpy


# A bond in TN is a bridge :<=> when remove the bond, the TN is disconnected into left and right
class BridgeBondEnv:
    def __init__(self,leftTensor,rightTensor,ket_left_labels,ket_right_labels):
        self.leftTensor = leftTensor
        self.rightTensor = rightTensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)

    def optimal_truncate(self, sigma0, chi=20):
        dl_label = unique_label()
        dr_label = unique_label()

        V_L = self.leftTensor
        V_R = self.rightTensor

        Yh, d_L, Y = tnd.truncated_eigh(V_L, self.ket_left_labels, self.bra_left_labels, eigh_labels=dl_label)
        Y.replace_labels(self.bra_left_labels, self.ket_left_labels, inplace=True)
        X, d_R, Xh = tnd.truncated_eigh(V_R, self.ket_right_labels, self.bra_right_labels, eigh_labels=dr_label)
        Xh.replace_labels(self.bra_right_labels, self.ket_right_labels, inplace=True)
        # if truncated, it means "no sufficient terms to decide M,S,N", so that 
        # ((M*S*N)-sigma0).norm() != 0
        # (((M*S*N)-sigma0)*H_L*H_R).norm() == 0 (ETA_L=H_L*H_L.adjoint, ETA_R=H_R*H_R.adjoint)

        G = d_L.sqrt() * Yh * sigma0 * X * d_R.sqrt()
        U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi)
        M = Y * d_L.sqrt().inv() * U
        N = V * d_R.sqrt().inv() * Xh

        return M, S, N





class UnbridgeBondEnv:
    def __init__(self,tensor,ket_left_labels,ket_right_labels):
        self.tensor = tensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)


    def optimal_truncate(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, memo=None):
        if memo is None:
            memo = {}
        start_time = time.time()

        maxchi = max(1,chi)
        del chi
        b = soujou(sigma0.dims(self.ket_left_labels))
        memo["b"] = b
        n = numpy.linalg.matrix_rank(self.tensor.to_matrix(self.ket_left_labels+self.ket_right_labels))
        memo["n"] = n
        avoiding_singular_chi = n // b
        exactly_solvable_chi = math.ceil((4*b+1 - math.sqrt((4*b+1)**2-8*b-16*n))/4)

        if exactly_solvable_chi <= maxchi:
            chi = exactly_solvable_chi
            memo["exactly_solvable"] = True
        else:
            chi = maxchi
            memo["exactly_solvable"] = False

        ket_mn_label = unique_label()
        bra_mn_label = aster_label(ket_mn_label)

        ETA = self.tensor
        Cbase = ETA * sigma0

        M,S,N = tnd.truncated_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = ket_mn_label)
        M = M * S.sqrt()
        N = S.sqrt() * N
        del S


        if n <= chi*b:
            # assert memo["exactly_solvable"] # proven in test0167
            memo["used_algorithm"] = "exact_solving"
            # solve exactly
            # this case includes avoiding_singular_chi==0 case. (Because if n<b: exactly_solvable_chi==1).

            extraction_label = unique_label()
            HTA,HTAw,_ = tnd.truncated_eigh(ETA, self.ket_left_labels+self.ket_right_labels, chi=chi*b, atol=0, rtol=0, eigh_labels=extraction_label) #TODO sometimes segmentation fault occurs (why?)
            #print(HTAw)
            Mshape = M.shape
            B = N * HTA
            B = B.to_matrix(extraction_label, self.ket_left_labels+[ket_mn_label])
            C = sigma0 * HTA
            C = C.to_vector(extraction_label)
            M = xp.linalg.solve(B,C)
            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_mn_label])
            #print(M*N*HTA - sigma0*HTA)

            memo["iter_times"] = 0


        else: # chi < n/b  ==>  chi*b<n  ==>  B is not singular
            # assert chi <= avoiding_singular_chi # proven in test0167
            memo["used_algorithm"] = "iterating_method"

            def optimize_M_from_N(N):
                Nh = N.adjoint(ket_mn_label, self.ket_right_labels, style="aster")
                B = N * ETA * Nh
                C = Cbase * Nh
                Mshape = B.dims(self.ket_left_labels+[ket_mn_label])
                B = B.to_matrix(self.bra_left_labels+[bra_mn_label], self.ket_left_labels+[ket_mn_label])
                C = C.to_vector(self.bra_left_labels+[bra_mn_label])
                M = xp.linalg.solve(B, C, assume_a="pos")

                M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_mn_label])
                return M

            def optimize_N_from_M(M):
                Mh = M.adjoint(self.ket_left_labels, ket_mn_label, style="aster")
                B = Mh * ETA * M
                C = Mh * Cbase
                Nshape = B.dims([ket_mn_label]+self.ket_right_labels)
                B = B.to_matrix([bra_mn_label]+self.bra_right_labels, [ket_mn_label]+self.ket_right_labels)
                C = C.to_vector([bra_mn_label]+self.bra_right_labels)
                N = xp.linalg.solve(B, C, assume_a="pos")
                N = tnc.vector_to_tensor(N, Nshape, [ket_mn_label]+self.ket_right_labels)
                return N

            for iteri in range(maxiter):
                oldM = M
                M = optimize_M_from_N(N)
                N = optimize_N_from_M(M)
                if M.__eq__(oldM, atol=conv_atol, rtol=conv_rtol):
                    break
            
            memo["iter_times"] = iteri


        M,S,N = tnd.truncated_svd(M*N, self.ket_left_labels, self.ket_right_labels, chi=chi)

        memo["chi"] = chi
        memo["elapsed_time"] = time.time()-start_time


        return M,S,N
