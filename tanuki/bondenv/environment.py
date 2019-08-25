from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



# If a bond in TN is a bridge, when remove the bond, the TN is disconnected into left and right
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

        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.ket_left_labels, self.bra_left_labels, eigh_labels=dl_label)
        Y.replace_labels(self.bra_left_labels, self.ket_left_labels, inplace=True)
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.ket_right_labels, self.bra_right_labels, eigh_labels=dr_label)
        Xh.replace_labels(self.bra_right_labels, self.ket_right_labels, inplace=True)

        G = d_L.sqrt() * Yh * sigma0 * X * d_R.sqrt()
        U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi, rtol=rtol, atol=atol)
        M = Y * d_L.inv().sqrt() * U
        N = V * d_R.inv().sqrt() * Xh

        return M, S, N



class UnbridgeBondEnv:
    def __init__(self,tensor,ket_left_labels,ket_right_labels):
        self.tensor = tensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = aster_labels(ket_left_labels)
        self.bra_right_labels = aster_labels(ket_right_labels)


    def optimal_truncate(self, sigma0, chi=20):
        ket_ms_label = unique_label()
        ket_sn_label = unique_label()
        bra_ms_label = aster_label(ket_ms_label)
        bra_sn_label = aster_label(ket_sn_label)

        ETA = self.tensor

        def optimize_M_from_S_N(S,N):
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            Nh = N.adjoint(ket_sn_label, self.ket_right_label, style="aster")
            B = ETA * Nh * Sh
            C = B * sigma0
            B = S * N * B
            Mshape = B.dims(self.ket_left_labels+[ket_ms_label])
            B = B.to_matrix(self.bra_left_labels+[bra_ms_label], self.ket_left_labels+[ket_ms_label])
            C = C.to_vector(self.bra_left_labels+[bra_ms_label])
            M = xp.linalg.solve(B, C)
            M = tnc.vector_to_tensor(M, Mshape, self.ket_left_labels+[ket_ms_label])
            return M

        def optimize_N_from_M_S(M,S):
            Mh = M.adjoint(self.ket_left_label, ket_ms_label, style="aster")
            Sh = S.adjoint(ket_ms_label, ket_sn_label, style="aster")
            B = Sh * Mh * ETA
            C = B * sigma0
            B = B * M * S
            Nshape = B.dims([ket_sn_label]+self.ket_right_labels)
            B = B.to_matrix([bra_sn_label]+self.bra_right_labels, [ket_sn_label]+self.ket_right_labels)
            C = C.to_vector([bra_sn_label]+self.bra_right_labels)
            N = xp.linalg.solve(B, C)
            N = tnc.vector_to_tensor(N, Nshape, [ket_sn_label]+self.ket_right_labels)
            return N

        M,S,N = tnc.truncated_svd(sigma0, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])

        for _ in range(20):
            M = optimize_M_from_S_N(S,N)
            N = optimize_N_from_M_S(M,S)
            M,S,N = tnc.truncated_svd(M*S*N, self.ket_left_labels, self.ket_right_labels, chi=chi, svd_labels = [ket_ms_label, ket_sn_label])

        return M,S,N
