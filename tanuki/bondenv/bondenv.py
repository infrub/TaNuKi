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
class BridgeBondOptimalTruncator:
    def __init__(self, *args, **kwargs):
        self.params = {"chi":20}
        if len(args) >= 4:
            self.set_env(*args[:4])
        if len(args) >= 5:
            self.set_A0(args[4])
        self.set_params(**kwargs)

    def set_env(self,leftTensor,rightTensor,ket_left_labels,ket_right_labels):
        self.leftTensor = leftTensor
        self.rightTensor = rightTensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)

    def set_A0(self,A0):
        self.A0 = A0

    def set_params(self, **kwargs):
        self.params.update(kwargs)

    def initialize(self):
        self.chi = self.params["chi"]

    def run(self):
        self.initialize()
        dl_label = unique_label()
        dr_label = unique_label()

        V_L = self.leftTensor
        V_R = self.rightTensor

        Yh, d_L, Y = tnd.truncated_eigh(V_L, self.ket_left_labels, self.bra_left_labels, eigh_labels=dl_label)
        Y.replace_labels(self.bra_left_labels, self.ket_left_labels, inplace=True)
        X, d_R, Xh = tnd.truncated_eigh(V_R, self.ket_right_labels, self.bra_right_labels, eigh_labels=dr_label)
        Xh.replace_labels(self.bra_right_labels, self.ket_right_labels, inplace=True)
        # if truncated, it means "no sufficient terms to decide M,S,N", so that 
        # ((M*S*N)-A0).norm() != 0
        # (((M*S*N)-A0)*H_L*H_R).norm() == 0 (ETA_L=H_L*H_L.adjoint, ETA_R=H_R*H_R.adjoint)

        G = d_L.sqrt() * Yh * self.A0 * X * d_R.sqrt()
        P, S, Q = tnd.tensor_svd(G, dl_label, dr_label, chi=self.chi)
        U = Y * d_L.sqrt().inv() * P
        V = Q * d_R.sqrt().inv() * Xh

        self.U = U
        self.S = S
        self.V = V

        return U,S,V



class UnbridgeBondOptimalTruncator:
    def __init__(self,*args,**kwargs):
        params = {}
        params["chi"] = 20
        params["max_iter"] = 1000
        params["conv_atol"] = 1e-10
        params["conv_rtol"] = 1e-10
        params["conv_sqdiff"] = -float("inf")
        params["algname"] = "NOR"
        params["linalg_algname"] = "solve"
        self.params = params
        if len(args) >= 3:
            self.set_env(*args[:3])
        if len(args) >= 4:
            self.set_A0(args[3])
        self.set_params(**kwargs)

    def __str__(self):
        return ""

    def set_env(self,ETA,ket_left_labels,ket_right_labels):
        self.ETA = ETA
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)

    def set_A0(self,A0):
        self.A0 = A0

    def set_params(self, **kwargs):
        self.params.update(kwargs)


    def initialize(self):
        # Judge whether need exact_solving
        b = soujou(self.A0.dims(self.ket_left_labels))
        self.b = b
        n = numpy.linalg.matrix_rank(self.ETA.to_matrix(self.ket_left_labels+self.ket_right_labels), hermitian=True)
        self.n = n
        max_chi_can_use_iterating_method = n // b # = floor(n/b)
        min_chi_can_use_exact_solving = (n-1)//b+1 # = ceil(n/b)
        exactly_solvable_chi = math.ceil(b - math.sqrt(b**2-n))

        chi = self.params["chi"]
        if chi >= min_chi_can_use_exact_solving:
            chi = min_chi_can_use_exact_solving
            #assert self.params["has_enough_degree_of_freedom_to_solve_exactly"] #proven
            algname = "exact_solving"
        else:
            #assert chi <= max_chi_can_use_iterating_method #proven
            algname = self.params["algname"]

        self.has_enough_degree_of_freedom_to_solve_exactly = (4*b*chi - 2*chi*chi) >= 2*n
        self.chi = chi
        self.algname = algname
        self.linalg_algname = self.params["linalg_algname"]

        self.ket_mn_label = unique_label()
        self.bra_mn_label = aster_label(self.ket_mn_label)

        U,S,V = tnd.tensor_svd(self.A0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = self.ket_mn_label)
        M = U * S.sqrt()
        N = S.sqrt() * V

        self._U = U
        self._S = S
        self._V = V
        self.M = M
        self.N = N
        self.last_USV_calced_iter_times = 0

        self.Cbase = self.ETA * self.A0
        self.sqdiff = ((M*N-self.A0)*self.ETA*(M*N-self.A0).adjoint()).real().to_scalar()
        self.sqdiff_history = [self.sqdiff]
        self.iter_times = 0

        if algname in ["WCOR","SCOR"]:
            if "omega" not in self.params:
                self.params["omega"] = min(1.95,1.2+0.125*np.log(self.b*self.chi))
        elif algname in ["WROR","SROR","IWROR"]:
            if "omega_cands" not in self.params:
                self.params["omega_cands"] = [1.6,1.65,1.7,1.72,1.74,1.76,1.78,1.8,1.85,1.94,1.95]
        elif algname in ["WMSGD", "WNAG", "SMSGD", "SNAG"]:
            if "emamu" not in self.params:
                self.params["emamu"] = 0.9
            self.lastDM = None
            self.lastDN = None
        elif algname in ["SSpiral"]:
            if "spiral_turn_max" not in self.params:
                self.params["spiral_turn_max"] = self.b


    def run_onestep(self):
        def optimize_M_from_N(M,N):
            Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
            B = N * self.ETA * Nh
            C = self.Cbase * Nh
            Mshape = M.dims(self.ket_left_labels+[self.ket_mn_label])
            B = B.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
            C = C.to_vector(self.bra_left_labels+[self.bra_mn_label])
            if self.linalg_algname == "solve":
                M = xp.linalg.solve(B, C, assume_a="pos")
            elif self.linalg_algname == "cg":
                M = M.to_vector(self.ket_left_labels+[self.ket_mn_label])
                M,_ = xp.sparse.linalg.cg(B,C,M,tol=0,max_iter=kwargs.get("linalg_max_iter",self.b*self.chi))
            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[self.ket_mn_label])
            return M

        def optimize_N_from_M(M,N):
            Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
            B = Mh * self.ETA * M
            C = Mh * self.Cbase
            Nshape = N.dims([self.ket_mn_label]+self.ket_right_labels)
            B = B.to_matrix([self.bra_mn_label]+self.bra_right_labels, [self.ket_mn_label]+self.ket_right_labels)
            C = C.to_vector([self.bra_mn_label]+self.bra_right_labels)
            if self.linalg_algname == "solve":
                N = xp.linalg.solve(B, C, assume_a="pos")
            elif self.linalg_algname == "cg":
                N = N.to_vector([self.ket_mn_label]+self.ket_right_labels)
                N,_ = xp.sparse.linalg.cg(B,C,N,tol=0,max_iter=kwargs.get("linalg_max_iter",self.b*self.chi))
            N = tnc.vector_to_tensor(N, Nshape, [self.ket_mn_label]+self.ket_right_labels)
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
            Tss[0][0] = M*N-self.A0
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
                            css[i+j][k+l] += (Tss[i][k] * self.ETA * Thss[j][l]).real().to_scalar()

            if loglevel>=20: 
                print("[",end="")
                for i in range(3):
                    print("[",end="")
                    for k in range(3):
                        print(f"{css[i][k]:+.4e}",end=" ")
                    print("]",end="")
                print("]",end="")

            return css



        M = self.M
        N = self.N
        if self.algname in ["WCOR","SCOR"]:
            omega = self.params["omega"]
        elif self.algname in ["WROR","SROR","IWROR"]:
            omega_cands = self.params["omega_cands"]
        elif self.algname in ["WMSGD", "WNAG", "SMSGD", "SNAG"]:
            emamu = self.params["emamu"]
            lastDM = self.lastDM
            lastDN = self.lastDN
        elif self.algname in ["SSpiral"]:
            spiral_turn_max = self.params["spiral_turn_max"]


        if self.algname == "NOR": # no over-relaxation
            oldM = M
            M = optimize_M_from_N(M,N)
            N = optimize_N_from_M(M,N)

        elif self.algname == "WCOR": # constant over-relaxation
            oldM = M
            stM = optimize_M_from_N(M,N)
            M = stM*omega - (omega-1)*M
            stN = optimize_N_from_M(M,N)
            N = stN*omega - (omega-1)*N

        elif self.algname == "WLBOR": # local best over-relaxation
            oldM = M
            stM = optimize_M_from_N(M,N)
            stN = optimize_N_from_M(stM,N)
            dM = stM - M
            dN = stN - N
            x,fx = solve_argmin_xxxx_equation(css_to_cs(get_equation_coeffs(M,N,dM,dN)))
            M = M + x*dM
            N = N + x*dN

        elif self.algname == "WROR": # randomized over-relaxation
            omega = random.choice(omega_cands)
            oldM = M
            stM = optimize_M_from_N(M,N)
            M = stM*omega - (omega-1)*M
            stN = optimize_N_from_M(M,N)
            N = stN*omega - (omega-1)*N

        elif self.algname == "IWROR": # individually randomized over-relaxation
            oldM = M
            omega = random.choice(omega_cands)
            stM = optimize_M_from_N(M,N)
            M = stM*omega - (omega-1)*M
            omega = random.choice(omega_cands)
            stN = optimize_N_from_M(M,N)
            N = stN*omega - (omega-1)*N

        elif self.algname == "WMSGD": # momentum
            stM = optimize_M_from_N(M,N)
            if lastDM is None:
                dM = (stM - M)
            else:
                dM = emamu * lastDM + (1-emamu)*(stM - M)
            M = stM + dM
            lastDM = dM

            stN = optimize_N_from_M(M,N)
            if lastDN is None:
                dN = (stN - N)
            else:
                dN = emamu * lastDN + (1-emamu)*(stN - N)
            N = stN + dN
            lastDN = dN

        elif self.algname == "WNAG":
            if lastDM is None:
                stM = optimize_M_from_N(M,N)
                dM = (stM - M)
            else:
                stM = optimize_M_from_N(M+emamu*lastDM, N)
                dM = emamu * lastDM + (1-emamu)*(stM - M)
            M = stM + dM
            lastDM = dM

            if lastDN is None:
                stN = optimize_N_from_M(M,N)
                dN = (stN - N)
            else:
                stN = optimize_N_from_M(M, N+emamu*lastDN)
                dN = emamu * lastDN + (1-emamu)*(stN - N)
            N = stN + dN
            lastDN = dN

        elif self.algname == "SCOR": # constant over-relaxation
            oldM = M
            stM = optimize_M_from_N(M,N)
            M = stM*omega - (omega-1)*M
            N = optimize_N_from_M(M,N)

        elif self.algname == "SROR": # randomized over-relaxation
            omega = random.choice(omega_cands)
            oldM = M
            stM = optimize_M_from_N(M,N)
            M = stM*omega - (omega-1)*M
            N = optimize_N_from_M(M,N)

        elif self.algname == "SMSGD": # momentum
            stM = optimize_M_from_N(M,N)
            if lastDM is None:
                dM = (stM - M)
            else:
                dM = emamu * lastDM + (1-emamu)*(stM - M)
            M = stM + dM
            lastDM = dM

            N = optimize_N_from_M(M,N)

        elif self.algname == "SNAG":
            if lastDM is None:
                stM = optimize_M_from_N(M,N)
                dM = (stM - M)
            else:
                stM = optimize_M_from_N(M+emamu*lastDM, N)
                dM = emamu * lastDM + (1-emamu)*(stM - M)
            M = stM + dM
            lastDM = dM

            N = optimize_N_from_M(M,N)

        elif self.algname == "SSpiral":
            M0 = M
            M = optimize_M_from_N(M,N)
            N = optimize_N_from_M(M,N)
            M1 = M
            dM1 = (M1 - M0).data
            M = optimize_M_from_N(M,N)
            N = optimize_N_from_M(M,N)
            M2 = M
            dM2 = (M2 - M1).data
            # [abs(dM2/dM1) < 1] dM1*dM1/(dM1-dM2) = dM1 * 1/(1-dM2/dM1)
            # [else] dM1 * 10 = dM1 * 1/(1-0.9)
            temp = dM2 / dM1
            temp[abs(temp) >= 1] = 1-(1/spiral_turn_max)
            sdMfin = dM1 / (1 - temp)
            M.data = M0.data + sdMfin
            N = optimize_N_from_M(M,N)

        else:
            raise ArgumentError(f"no such algname == {self.algname}")


        self.M = M
        self.N = N
        if self.algname in ["WMSGD", "WNAG", "SMSGD", "SNAG"]:
            self.lastDM = lastDM
            self.lastDN = lastDN

        sqdiff = ((M*N-self.A0)*self.ETA*(M*N-self.A0).adjoint()).real().to_scalar()
        self.sqdiff_history.append(sqdiff)
        self.sqdiff = sqdiff
        self.iter_times += 1


    def run(self):
        start_time = time.time()
        self.initialize()

        if self.algname == "exact_solving":
            # solve exactly
            # this case includes max_chi_can_use_iterating_method==0 case. (Because if n<b: exactly_solvable_chi==1).

            M = self.M
            N = self.N
            extraction_label = unique_label()
            HTA,HTAw,_ = tnd.truncated_eigh(self.ETA, self.ket_left_labels+self.ket_right_labels, chi=self.chi*self.b, atol=0, rtol=0, eigh_labels=extraction_label) #TODO sometimes segmentation fault occurs (why?)
            #print(HTAw)
            Mshape = M.shape
            B = N * HTA
            B = B.to_matrix(extraction_label, self.ket_left_labels+[self.ket_mn_label])
            C = self.A0 * HTA
            C = C.to_vector(extraction_label)
            M = xp.linalg.solve(B,C)
            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[self.ket_mn_label])
            #print(M*N*HTA - A0*HTA)
            self.M = M
            self.N = N
            self.iter_times += 1

        else: # chi < n/b  ==>  chi*b<n  ==>  B is not singular
            #assert chi <= max_chi_can_use_iterating_method # proven in test0167

            while self.iter_times < self.params["max_iter"]:
                old_sqdiff = self.sqdiff
                self.run_onestep()
                if abs(self.sqdiff-old_sqdiff) <= self.sqdiff*self.params["conv_rtol"] + self.params["conv_atol"]:
                    break
                if self.sqdiff <= self.params["conv_sqdiff"]:
                    break

        self.elapsed_time = time.time()-start_time

        return self.U, self.S, self.V



    def recalc_USV(self):
        if self.last_USV_calced_iter_times != self.iter_times:
            self._U,self._S,self._V = tnd.tensor_svd(self.M * self.N, self.ket_left_labels, self.ket_right_labels, chi=self.chi, svd_labels = self.ket_mn_label)
            self.last_USV_calced_iter_times = self.iter_times

    @property
    def U(self):
        self.recalc_USV()
        return self._U

    @property
    def S(self):
        self.recalc_USV()
        return self._S
    
    @property
    def V(self):
        self.recalc_USV()
        return self._V

    @property
    def grad_by_reM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        re1 = N * (self.ETA * (Mh * Nh - A0h))
        re1 = re1.to_vector(self.ket_left_labels+[self.ket_mn_label])
        re2 = (M*N - A0) * self.ETA * Nh
        re2 = re2.to_vector(self.bra_left_labels+[self.bra_mn_label])
        return xp.real(re1 + re2)
    
    @property
    def grad_by_imM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        re1 = 1.0j * N * (self.ETA * (Mh * Nh - A0h))
        re1 = re1.to_vector(self.ket_left_labels+[self.ket_mn_label])
        re2 = -1.0j * (M*N - A0) * self.ETA * Nh
        re2 = re2.to_vector(self.bra_left_labels+[self.bra_mn_label])
        return xp.real(re1 + re2)

    @property
    def grad_by_M(self):
        return xp.concatenate([self.grad_by_reM, self.grad_by_imM])

    @property
    def grad(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")

        K_dM = N * (self.ETA * (Mh * Nh - A0h))
        K_dN = M * (self.ETA * (Mh * Nh - A0h))
        K_dMh = (M * N - A0) * self.ETA * Nh
        K_dNh = (M * N - A0) * self.ETA * Mh

        G_dReM = K_dM.to_vector(self.ket_left_labels+[self.ket_mn_label]) + K_dMh.to_vector(self.bra_left_labels+[self.bra_mn_label])
        G_dImM = 1.0j * K_dM.to_vector(self.ket_left_labels+[self.ket_mn_label]) - 1.0j * K_dMh.to_vector(self.bra_left_labels+[self.bra_mn_label])
        G_dReN = K_dN.to_vector(self.ket_right_labels+[self.ket_mn_label]) + K_dNh.to_vector(self.bra_right_labels+[self.bra_mn_label])
        G_dImN = 1.0j * K_dN.to_vector(self.ket_right_labels+[self.ket_mn_label]) - 1.0j * K_dNh.to_vector(self.bra_right_labels+[self.bra_mn_label])

        parts = [xp.real(G_dReM), xp.real(G_dImM), xp.real(G_dReN), xp.real(G_dImN)]
        return xp.concatenate(parts)


    @property
    def hessian_by_reM_reM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        retemp = N * self.ETA * Nh
        re1 = retemp.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        re2 = retemp.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        return xp.real(re1 + re2) # motomoto real
    
    @property
    def hessian_by_imM_imM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        retemp = N * self.ETA * Nh
        re1 = retemp.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        re2 = retemp.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        return xp.real(re1 + re2) # motomoto real
    
    @property
    def hessian_by_reM_imM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        re1 = N * self.ETA * (-1.0j*Nh)
        re1 = re1.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        re2 = (1.0j*N) * self.ETA * Nh
        re2 = re2.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        return xp.real(re1 + re2) # motomoto real
    
    @property
    def hessian_by_imM_reM(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")
        re1 = (1.0j*N) * self.ETA * Nh
        re1 = re1.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        re2 = N * self.ETA * (-1.0j*Nh)
        re2 = re2.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        return xp.real(re1 + re2) # motomoto real

    @property
    def hessian_by_M_M(self):
        return xp.block([[self.hessian_by_reM_reM, self.hessian_by_reM_imM],[self.hessian_by_imM_reM, self.hessian_by_imM_imM]])

    @property
    def hessian(self):
        M = self.M
        N = self.N
        A0 = self.A0
        Mh = M.adjoint(self.ket_left_labels, self.ket_mn_label, style="aster")
        Nh = N.adjoint(self.ket_mn_label, self.ket_right_labels, style="aster")
        A0h = A0.adjoint(self.ket_left_labels, self.ket_right_labels, style="aster")

        K_dM_dN = self.ETA * (Mh*Nh - A0h) * tni.identity_diagonalTensor(M.dims(self.ket_mn_label), [self.ket_mn_label, self.ket_mn_label])
        K_dMh_dNh = (M*N - A0) * self.ETA * tni.identity_diagonalTensor(Mh.dims(self.bra_mn_label), [self.bra_mn_label, self.bra_mn_label])
        K_dM_dMh = N * self.ETA * Nh
        K_dM_dNh = N * self.ETA * Mh
        K_dN_dMh = M * self.ETA * Nh
        K_dN_dNh = M * self.ETA * Mh

        H = {"dReM":{},"dReN":{},"dImM":{},"dImN":{}}
        # H["dReM"]["dReN"][i,j] = \del_{ReM_i}\del_{ReN_j} f
        H["dReM"]["dReM"] = K_dM_dMh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label]) + K_dM_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        H["dImM"]["dImM"] = K_dM_dMh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label]) + K_dM_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        H["dReM"]["dImM"] = -1.0j * K_dM_dMh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label]) + 1.0j *  K_dM_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])
        H["dImM"]["dReM"] = 1.0j * K_dM_dMh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label]) - 1.0j *  K_dM_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label])

        H["dReN"]["dReN"] = K_dN_dNh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) + K_dN_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dImN"]["dImN"] = K_dN_dNh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) + K_dN_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dReN"]["dImN"] = -1.0j * K_dN_dNh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) + 1.0j *  K_dN_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dImN"]["dReN"] = 1.0j * K_dN_dNh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) - 1.0j *  K_dN_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])

        H["dReM"]["dReN"] = \
            + K_dM_dN.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.ket_right_labels+[self.ket_mn_label]) \
            + K_dMh_dNh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + K_dM_dNh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + K_dN_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dReM"]["dImN"] = \
            + 1.0j * K_dM_dN.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.ket_right_labels+[self.ket_mn_label]) \
            - 1.0j * K_dMh_dNh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            - 1.0j * K_dM_dNh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + 1.0j * K_dN_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dImM"]["dReN"] = \
            + 1.0j * K_dM_dN.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.ket_right_labels+[self.ket_mn_label]) \
            - 1.0j * K_dMh_dNh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + 1.0j * K_dM_dNh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            - 1.0j * K_dN_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])
        H["dImM"]["dImN"] = \
            - K_dM_dN.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.ket_right_labels+[self.ket_mn_label]) \
            - K_dMh_dNh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + K_dM_dNh.to_matrix(self.ket_left_labels+[self.ket_mn_label], self.bra_right_labels+[self.bra_mn_label]) \
            + K_dN_dMh.to_matrix(self.bra_left_labels+[self.bra_mn_label], self.ket_right_labels+[self.ket_mn_label])

        H["dReN"]["dReM"] = \
            + K_dM_dN.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            + K_dMh_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.bra_left_labels+[self.bra_mn_label]) \
            + K_dM_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            + K_dN_dMh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        H["dImN"]["dReM"] = \
            + 1.0j * K_dM_dN.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            - 1.0j * K_dMh_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.bra_left_labels+[self.bra_mn_label]) \
            - 1.0j * K_dM_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            + 1.0j * K_dN_dMh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        H["dReN"]["dImM"] = \
            + 1.0j * K_dM_dN.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            - 1.0j * K_dMh_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.bra_left_labels+[self.bra_mn_label]) \
            + 1.0j * K_dM_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            - 1.0j * K_dN_dMh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])
        H["dImN"]["dImM"] = \
            - K_dM_dN.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            - K_dMh_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.bra_left_labels+[self.bra_mn_label]) \
            + K_dM_dNh.to_matrix(self.bra_right_labels+[self.bra_mn_label], self.ket_left_labels+[self.ket_mn_label]) \
            + K_dN_dMh.to_matrix(self.ket_right_labels+[self.ket_mn_label], self.bra_left_labels+[self.bra_mn_label])

        partss = [[xp.real(H[a][b]) for b in ["dReM","dImM","dReN","dImN"]] for a in ["dReM","dImM","dReN","dImN"]] #douse real
        return xp.block(partss)


