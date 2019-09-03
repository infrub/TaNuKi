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


    def optimal_truncate(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, memo=None, algname="alg01"):
        if memo is None:
            memo = {}
        memo["algname"] = algname
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

            def solve_argmin_xxxx_equation(cs):
                # x = argmin( \sum_{i=0..4} x^i cs[i] ) # css is real, x is real.
                xkouhos = numpy.roots([4*cs[4], 3*cs[3], 2*cs[2], 1*cs[1]])
                x = 0
                fx = cs[0]
                for xkouho in xkouhos: #TODO totsu nanjanaika? dattara for iranaiyone
                    if np.iscomplex(xkouho): continue
                    fxkouho = cs[0] + cs[1] * xkouho + cs[2] * xkouho**2 + cs[3] * xkouho**3 + cs[4] * xkouho**4
                    if fxkouho < fx:
                        fx = fxkouho
                        x = xkouho
                return x

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

                cs = [0,0,0,0,0]
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                cs[i+j+k+l] += (Tss[i][k] * ETA * Thss[j][l]).real().to_scalar()

                return cs


            if algname == "alg01":
                for iteri in range(maxiter):
                    oldM = M
                    M = optimize_M_from_N(N)
                    N = optimize_N_from_M(M)
                    if M.__eq__(oldM, atol=conv_atol, rtol=conv_rtol):
                        break

            elif algname == "alg02": # sindou suru. kuso osoi.
                for iteri in range(maxiter):
                    oldM = M
                    M = optimize_M_from_N(N)
                    N = optimize_N_from_M(oldM)
                    if M.__eq__(oldM, atol=conv_atol, rtol=conv_rtol):
                        break

            elif algname == "alg03": # musiro osoi.
                for iteri in range(maxiter):
                    kariNewM = optimize_M_from_N(N)
                    kariNewN = optimize_N_from_M(M)
                    if M.__eq__(kariNewM, atol=conv_atol, rtol=conv_rtol):
                        break
                    dM = kariNewM - M
                    dN = kariNewN - N
                    x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                    if x==0:
                        break
                    M = M + x*dM
                    N = N + x*dN
            
            elif algname == "alg04": # hayame
                for iteri in range(maxiter):
                    kariNewM = optimize_M_from_N(N)
                    kariNewN = optimize_N_from_M(kariNewM)
                    if M.__eq__(kariNewM, atol=conv_atol, rtol=conv_rtol):
                        break
                    dM = kariNewM - M
                    dN = kariNewN - N
                    x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                    if x==0:
                        break
                    M = M + x*dM
                    N = N + x*dN

            elif algname == "alg05": # gomikasu. alg02 de sindou surunoni ataru toki, dM,dN majide kankei nai houkou ni ikou to suru
                for iteri in range(maxiter):
                    kariNewM = optimize_M_from_N(optimize_N_from_M(M))
                    kariNewN = optimize_N_from_M(optimize_M_from_N(N))
                    if M.__eq__(optimize_M_from_N(N), atol=conv_atol, rtol=conv_rtol):
                        break
                    dM = kariNewM - M
                    dN = kariNewN - N
                    x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                    #if x==0:
                    #    break
                    M = M + x*dM
                    N = N + x*dN

            elif algname == "alg06": # opt N no atoni opt N siteru wakedakara musiro osoi. alg03 yoriha tyoimasi?
                for iteri in range(maxiter):
                    if iteri % 2 == 0:
                        kariNewM = optimize_M_from_N(N)
                        kariNewN = optimize_N_from_M(kariNewM)
                        if M.__eq__(kariNewM, atol=conv_atol, rtol=conv_rtol):
                            break
                        dM = kariNewM - M
                        dN = kariNewN - N
                        x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                        if x==0:
                            break
                        M = M + x*dM
                        N = N + x*dN
                    else:
                        kariNewN = optimize_N_from_M(M)
                        kariNewM = optimize_M_from_N(kariNewN)
                        if N.__eq__(kariNewN, atol=conv_atol, rtol=conv_rtol):
                            break
                        dM = kariNewM - M
                        dN = kariNewN - N
                        x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                        if x==0:
                            break
                        M = M + x*dM
                        N = N + x*dN

            elif algname == "alg07":
                for iteri in range(maxiter):
                    if iteri % 2 == 0:
                        kariNewM = optimize_M_from_N(N)
                        kariNewN = optimize_N_from_M(kariNewM)
                        if M.__eq__(kariNewM, atol=conv_atol, rtol=conv_rtol):
                            break
                        dM = kariNewM - M
                        dN = kariNewN - N
                        x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                        if x==0:
                            break
                        M = M + x*dM
                    else:
                        kariNewN = optimize_N_from_M(M)
                        kariNewM = optimize_M_from_N(kariNewN)
                        if N.__eq__(kariNewN, atol=conv_atol, rtol=conv_rtol):
                            break
                        dM = kariNewM - M
                        dN = kariNewN - N
                        x = solve_argmin_xxxx_equation(get_equation_coeffs(M,N,dM,dN))
                        if x==0:
                            break
                        N = N + x*dN


            memo["iter_times"] = iteri


        M,S,N = tnd.truncated_svd(M*N, self.ket_left_labels, self.ket_right_labels, chi=chi)

        memo["elapsed_time"] = time.time()-start_time


        return M,S,N




