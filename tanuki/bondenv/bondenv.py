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


    def optimal_truncate_alg01(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, memo=None):
        if memo is None:
            memo = {}
        start_time = time.time()

        chi = max(1,chi)
        b = soujou(sigma0.dims(self.ket_left_labels))
        memo["b"] = b
        n = numpy.linalg.matrix_rank(self.tensor.to_matrix(self.ket_left_labels+self.ket_right_labels), hermitian=True)
        memo["n"] = n
        max_chi_can_use_iterating_method = n // b # = floor(n/b)
        min_chi_can_use_exact_solving = (n-1)//b+1 # = ceil(n/b)
        exactly_solvable_chi = math.ceil(b - math.sqrt(b**2-n))

        if chi >= min_chi_can_use_exact_solving:
            chi = min_chi_can_use_exact_solving
            #assert memo["has_enough_degree_of_freedom_to_solve_exactly"] #proven
            memo["used_algorithm"] = "exact_solving"
        else:
            #assert chi <= max_chi_can_use_iterating_method #proven
            memo["used_algorithm"] = "iterating_method"
        memo["chi"] = chi
        memo["has_enough_degree_of_freedom_to_solve_exactly"] = (4*b*chi - 2*chi*chi) >= 2*n


        ket_mn_label = unique_label()
        bra_mn_label = aster_label(ket_mn_label)

        ETA = self.tensor
        Cbase = ETA * sigma0

        M,S,N = tnd.truncated_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = ket_mn_label)
        M = M * S.sqrt()
        N = S.sqrt() * N
        del S


        if memo["used_algorithm"] == "exact_solving":
            # solve exactly
            # this case includes max_chi_can_use_iterating_method==0 case. (Because if n<b: exactly_solvable_chi==1).

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
            assert chi <= max_chi_can_use_iterating_method # proven in test0167
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

        memo["elapsed_time"] = time.time()-start_time


        return M,S,N



    def optimal_truncate_alg02(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, memo=None):
        if memo is None:
            memo = {}
        start_time = time.time()

        chi = max(1,chi)
        b = soujou(sigma0.dims(self.ket_left_labels))
        memo["b"] = b
        n = numpy.linalg.matrix_rank(self.tensor.to_matrix(self.ket_left_labels+self.ket_right_labels), hermitian=True)
        memo["n"] = n
        max_chi_can_use_iterating_method = n // b # = floor(n/b)
        min_chi_can_use_exact_solving = (n-1)//b+1 # = ceil(n/b)
        exactly_solvable_chi = math.ceil(b - math.sqrt(b**2-n))

        if chi >= min_chi_can_use_exact_solving:
            chi = min_chi_can_use_exact_solving
            #assert memo["has_enough_degree_of_freedom_to_solve_exactly"] #proven
            memo["used_algorithm"] = "exact_solving"
        else:
            #assert chi <= max_chi_can_use_iterating_method #proven
            memo["used_algorithm"] = "iterating_method"
        memo["chi"] = chi
        memo["has_enough_degree_of_freedom_to_solve_exactly"] = (4*b*chi - 2*chi*chi) >= 2*n


        ket_mn_label = unique_label()
        bra_mn_label = aster_label(ket_mn_label)

        ETA = self.tensor
        Cbase = ETA * sigma0

        M,S,N = tnd.truncated_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = ket_mn_label)
        M = M * S.sqrt()
        N = S.sqrt() * N
        del S


        if memo["used_algorithm"] == "exact_solving":
            # solve exactly
            # this case includes max_chi_can_use_iterating_method==0 case. (Because if n<b: exactly_solvable_chi==1).

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
            assert chi <= max_chi_can_use_iterating_method # proven in test0167
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

            def solve_argmin_xxyy_equation(css):
                # x,y = argmin( \sum_{i=0,1,2}{j=0,1,2} x^i y^j css[i,j] ) # css is real, s,t is real.
                return 1,1

            def get_equation_coeffs(M,N,dM,dN):
                Tss = [[0,0],[0,0]]
                Tss[0][0] = M*N-sigma0
                Tss[1][0] = dM * N
                Tss[0][1] = M * dN
                Tss[1][1] = dM * dN
                Thss = [[0,0],[0,0]]
                for i in range(2):
                    for j in range(2):
                        Thss[i][j] = Tss[i][j].adjoint()
                # minimize_x,y || \sum_{i=0,1}{j=0,1} x^i y^j T[i,j] * HETA ||^2

                css = [[0,0,0],[0,0,0],[0,0,0]]
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                css[i+j][k+l] += (Tss[i][k] * ETA * Thss[j][l]).real().to_scalar()

                x,y = solve_argmin_xxyy_equation(css)

                print(css) # aruteido ikuto taishou gyouretu ni naru! uresii!

                return css


            for iteri in range(maxiter):
                kariNewM = optimize_M_from_N(N)
                kariNewN = optimize_N_from_M(M)
                if M.__eq__(kariNewM, atol=conv_atol, rtol=conv_rtol):
                    break
                dM = kariNewM - M
                dN = kariNewN - N
                x,y = solve_argmin_xxyy_equation(get_equation_coeffs(M,N,dM,dN))
                M = M + x*dM #temp
                N = N + y*dN #temp
                print(dM) # naruhodo! sindou siteru kara osoinone!
            
            memo["iter_times"] = iteri


        M,S,N = tnd.truncated_svd(M*N, self.ket_left_labels, self.ket_right_labels, chi=chi)

        memo["elapsed_time"] = time.time()-start_time


        return M,S,N



    optimal_truncate = optimal_truncate_alg01
