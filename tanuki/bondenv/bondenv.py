from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *
import warnings



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
        maxchi = soujou(sigma0.dims(self.ket_left_labels))
        chi = max(1,min(chi, maxchi))

        ket_ms_label = unique_label()
        ket_sn_label = unique_label()
        bra_ms_label = aster_label(ket_ms_label)
        bra_sn_label = aster_label(ket_sn_label)

        ETA = self.tensor
        Cbase = ETA * sigma0

        def optimize_M_from_S_N(S,N):
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            Nh = N.adjoint(ket_sn_label, self.ket_right_labels, style="aster")
            B = S * N * ETA * Nh * Sh
            C = Cbase * Nh * Sh
            Mshape = B.dims(self.ket_left_labels+[ket_ms_label])
            #assert B.is_hermite(self.ket_left_labels+[ket_ms_label])
            B = B.to_matrix(self.bra_left_labels+[bra_ms_label], self.ket_left_labels+[ket_ms_label])
            C = C.to_vector(self.bra_left_labels+[bra_ms_label])
            M = xp.linalg.solve(B, C, assume_a="pos")

            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_ms_label])
            return M

        def optimize_N_from_M_S(M,S):
            Mh = M.adjoint(self.ket_left_labels, ket_ms_label, style="aster")
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            B = Sh * Mh * ETA * M * S
            C = Sh * Mh * Cbase
            Nshape = B.dims([ket_sn_label]+self.ket_right_labels)
            B = B.to_matrix([bra_sn_label]+self.bra_right_labels, [ket_sn_label]+self.ket_right_labels)
            C = C.to_vector([bra_sn_label]+self.bra_right_labels)
            N = xp.linalg.solve(B, C, assume_a="pos")
            N = tnc.vector_to_tensor(N, Nshape, [ket_sn_label]+self.ket_right_labels)
            return N

        M,S,N = tnd.truncated_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])

        env_is_crazy_degenerated = False
        for iteri in range(maxiter):
            oldS = S
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    M = optimize_M_from_S_N(S,N)
                    N = optimize_N_from_M_S(M,S)
                except (xp.linalg.LinAlgError, xp.linalg.misc.LinAlgWarning) as e:
                    # Let (b,chi)=M.shape, n = rank(ENV). (b=maxchi)
                    # When b*chi > n, LinAlgError("matrix is singular") or LinAlgWarning("Ill-conditioned matrix: result may not be accurate") occurs, it means "no sufficient terms to decide M,S,N" then "with more small chi I can optimize M,S,N", so deal by shrinking chi.
                    # "no sufficient terms to decide M,S,N" => "with more small chi I can optimize M,S,N" is proven.
                    # Note: converse proposition does NOT work! (so chi can be wasteful even when the program did not storm in this block)
                    # the proof written by infrub is in test0111. need publishing? #TODO)

                    # Therefore finally the result become
                    # ((M*S*N)-sigma0).norm() != 0
                    # (((M*S*N)-sigma0)*H).norm() == 0 (ETA=H*H.adjoint)
                    if chi == 1:
                        # When b > n.
                        env_is_crazy_degenerated = True
                        break
                    else:
                        chi -= 1
                        M.truncate_index(ket_ms_label,chi,inplace=True)
                        S.truncate_index(ket_ms_label,chi,inplace=True)
                        N.truncate_index(ket_sn_label,chi,inplace=True)
                        continue
            M,S,N = tnd.truncated_svd(M*S*N, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])
            if S.__eq__(oldS, atol=conv_atol, rtol=conv_rtol):
                break

        if env_is_crazy_degenerated:
            # When b>n, even if chi=1, S * N * ETA * Nh * Sh is singular. So need special treatment.
            # However fortunately, when b>=n, decomposition ETA=HTA*_*_ (HTA:Matrix(b^2,n))) can be done and HTA*(M*S*N-sigma0)==0 can be achieved only by once M-optimizing.
            # it is done by solve( Matrix(n, b), Vector(b) ), but the calling is scared as "not square!" by numpy, add waste element in HTA to make it solve( Matrix(b, b), Vector(b) ).
            extraction_label = unique_label()
            HTA,_,_ = tnd.truncated_eigh(ETA, self.ket_left_labels+self.ket_right_labels, chi=maxchi, atol=0, rtol=0, eigh_labels=extraction_label) #TODO sometimes segmentation fault occurs (why?)
            Mshape = M.shape
            B = S * N * HTA
            B = B.to_matrix(extraction_label, self.ket_left_labels+[ket_ms_label])
            C = sigma0 * HTA
            C = C.to_vector(extraction_label)
            M = xp.linalg.solve(B,C)
            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_ms_label])
            M,S,N = tnd.truncated_svd(M*S*N, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])


        if memo is None:
            memo = {}
        memo["iter_times"] = iteri
        memo["env_is_crazy_degenerated"] = env_is_crazy_degenerated
        memo["chi"] = chi


        return M,S,N

