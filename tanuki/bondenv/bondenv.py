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
import numpy,scipy
import random



loglevel = 10

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


    def optimal_truncate(self, sigma0, chi=20, maxiter=1000, conv_atol=1e-10, conv_rtol=1e-10, conv_sqdiff=-float("inf"), memo=None, algname="NOR", linalg_algname="solve", **kwargs):
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
            algname = "exact_solving"
        else:
            #assert chi <= max_chi_can_use_iterating_method #proven
            pass
        memo["algname"] = algname
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


        if algname == "exact_solving":
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

            def optimize_M_from_N(M,N):
                Nh = N.adjoint(ket_mn_label, self.ket_right_labels, style="aster")
                B = N * ETA * Nh
                C = Cbase * Nh
                Mshape = M.dims(self.ket_left_labels+[ket_mn_label])
                B = B.to_matrix(self.bra_left_labels+[bra_mn_label], self.ket_left_labels+[ket_mn_label])
                C = C.to_vector(self.bra_left_labels+[bra_mn_label])
                if linalg_algname == "solve":
                    M = xp.linalg.solve(B, C, assume_a="pos")
                elif linalg_algname == "cg":
                    M = M.to_vector(self.ket_left_labels+[ket_mn_label])
                    M,_ = xp.sparse.linalg.cg(B,C,M,tol=0,maxiter=kwargs.get("linalg_maxiter",b*chi))
                M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_mn_label])
                return M

            def optimize_N_from_M(M,N):
                Mh = M.adjoint(self.ket_left_labels, ket_mn_label, style="aster")
                B = Mh * ETA * M
                C = Mh * Cbase
                Nshape = N.dims([ket_mn_label]+self.ket_right_labels)
                B = B.to_matrix([bra_mn_label]+self.bra_right_labels, [ket_mn_label]+self.ket_right_labels)
                C = C.to_vector([bra_mn_label]+self.bra_right_labels)
                if linalg_algname == "solve":
                    N = xp.linalg.solve(B, C, assume_a="pos")
                elif linalg_algname == "cg":
                    N = N.to_vector([ket_mn_label]+self.ket_right_labels)
                    N,_ = xp.sparse.linalg.cg(B,C,N,tol=0,maxiter=kwargs.get("linalg_maxiter",b*chi))
                N = tnc.vector_to_tensor(N, Nshape, [ket_mn_label]+self.ket_right_labels)
                return N

            def solve_argmin_xxxx_equation(cs):
                # x = argmin( \sum_{i=0..4} x^i cs[i] ) # css is real, x is real.
                xkouhos = numpy.roots([4*cs[4], 3*cs[3], 2*cs[2], 1*cs[1]])
                x = 0
                fx = cs[0]
                for xkouho in xkouhos: #TODO totsu nanjanaika? dattara for iranaiyone
                    if np.iscomplex(xkouho): continue
                    xkouho = np.real(xkouho)
                    fxkouho = cs[0] + cs[1] * xkouho + cs[2] * xkouho**2 + cs[3] * xkouho**3 + cs[4] * xkouho**4
                    if fxkouho < fx:
                        fx = fxkouho
                        x = xkouho
                if loglevel>=20: print(f"  {x:.10f},  {fx:.10e}")
                return x,fx

            def css_to_cs(css):
                cs = [0,0,0,0,0]
                for i in range(3):
                    for k in range(3):
                        cs[i+k] += css[i][k]
                return cs

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
                cs = [0,0,0,0,0]
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            for l in range(2):
                                css[i+j][k+l] += (Tss[i][k] * ETA * Thss[j][l]).real().to_scalar()

                if loglevel>=20: 
                    print("[",end="")
                    for i in range(3):
                        print("[",end="")
                        for k in range(3):
                            print(f"{css[i][k]:+.4e}",end=" ")
                        print("]",end="")
                    print("]",end="")

                return css


            oldFx = ((M*N-sigma0)*ETA*(M*N-sigma0).adjoint()).real().to_scalar()
            sqdiff_history = [oldFx]

            if algname == "COR":
                omega = kwargs.get("omega", min(1.95,1.2+0.125*np.log(b*chi)))
            elif algname == "ROR":
                omega_cands = kwargs.get("omega_cands",[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.85,1.94,1.95])
            elif algname == "IROR":
                omega_cands = kwargs.get("omega_cands",[1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.85,1.94,1.95])
                

            for iteri in range(maxiter):
                if algname == "NOR": # no over-relaxation
                    oldM = M
                    M = optimize_M_from_N(M,N)
                    N = optimize_N_from_M(M,N)
                    fx = ((M*N-sigma0)*ETA*(M*N-sigma0).adjoint()).real().to_scalar()

                elif algname == "COR": # constant over-relaxation
                    oldM = M
                    stM = optimize_M_from_N(M,N)
                    M = stM*omega - (omega-1)*M
                    stN = optimize_N_from_M(M,N)
                    N = stN*omega - (omega-1)*N
                    fx = ((stM*stN-sigma0)*ETA*(stM*stN-sigma0).adjoint()).real().to_scalar()

                elif algname == "LBOR": # local best over-relaxation
                    oldM = M
                    stM = optimize_M_from_N(M,N)
                    stN = optimize_N_from_M(stM,N)
                    dM = stM - M
                    dN = stN - N
                    x,fx = solve_argmin_xxxx_equation(css_to_cs(get_equation_coeffs(M,N,dM,dN)))
                    M = M + x*dM
                    N = N + x*dN
                    fx = ((stM*stN-sigma0)*ETA*(stM*stN-sigma0).adjoint()).real().to_scalar()

                elif algname == "ROR": # randomized over-relaxation
                    omega = random.choice(omega_cands)
                    oldM = M
                    stM = optimize_M_from_N(M,N)
                    M = stM*omega - (omega-1)*M
                    stN = optimize_N_from_M(M,N)
                    N = stN*omega - (omega-1)*N
                    fx = ((stM*stN-sigma0)*ETA*(stM*stN-sigma0).adjoint()).real().to_scalar()

                elif algname == "IROR": # individually randomized over-relaxation
                    oldM = M
                    omega = random.choice(omega_cands)
                    stM = optimize_M_from_N(M,N)
                    M = stM*omega - (omega-1)*M
                    omega = random.choice(omega_cands)
                    stN = optimize_N_from_M(M,N)
                    N = stN*omega - (omega-1)*N
                    fx = ((stM*stN-sigma0)*ETA*(stM*stN-sigma0).adjoint()).real().to_scalar()

                else:
                    raise Exception(f"no such algname == {algname}")

                sqdiff_history.append(fx)
                if abs(fx-oldFx) <= oldFx*conv_rtol + conv_atol:
                    print("tol conved",fx,oldFx,conv_rtol,conv_atol)
                    break
                if fx <= conv_sqdiff:
                    print("border conved")
                    break
                oldFx = fx



            memo["sqdiff_history"] = sqdiff_history
            memo["sqdiff"] = fx
            memo["iter_times"] = iteri+1


        M,S,N = tnd.truncated_svd(M*N, self.ket_left_labels, self.ket_right_labels, chi=chi)

        memo["elapsed_time"] = time.time()-start_time


        return M,S,N




