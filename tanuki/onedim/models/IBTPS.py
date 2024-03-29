from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.OBTPS import Opn1DBTPS

# ██ ██████  ████████ ██████  ███████ 
# ██ ██   ██    ██    ██   ██ ██      
# ██ ██████     ██    ██████  ███████ 
# ██ ██   ██    ██    ██           ██ 
# ██ ██████     ██    ██      ███████ 
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DBTPS(MixinInf1DBTP_, Opn1DBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        self.bdts = CyclicList(bdts)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Inf1DBTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

    def __str__(self, nodata=False):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += bdt.__str__(nodata=nodata)
                dataStr += "\n"
                tensor = self.tensors[i]
                dataStr += tensor.__str__(nodata=nodata)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Inf1DBTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self))]
        return Inf1DBTPS(tensors, bdts, self.phys_labelss)

    def to_tensor(self): # losts information
        re = 1
        for i in range(len(self)):
            re *= self.bdts[i]
            re *= self.tensors[i]
        return re

    def to_TPS(self):
        tensors = []
        for i in range(len(self)):
            tensors.append( self.bdts[i][self.get_right_labels_bond(i)] * self.tensors[i][self.get_left_labels_site(i)] )
        from tanuki.onedim.models.ITPS import Inf1DTPS
        return Inf1DTPS(tensors, self.phys_labelss)

    def to_BTPS(self):
        return self



    def equalize_norms(self, multiplier=1.0, divisor=1.0, normalize=False):
        sun = multiplier / divisor
        for i in range(len(self)):
            s = self.tensors[i].norm()
            self.tensors[i] = self.tensors[i] / s
            sun *= s
            s = self.bdts[i].norm()
            self.bdts[i] = self.bdts[i] / s
            sun *= s
        if normalize:
            return sun
        else:
            moon = sun ** (0.5/len(self))
            for i in range(len(self)):
                self.tensors[i] *= moon
                self.bdts[i] *= moon
            return 1.0

    def __imul__(self,other):
        if xp.isscalar(other):
            self.equalize_norms(multiplier=other)
            return self
        else:
            return NotImplemented

    def __itruediv__(self,other):
        if xp.isscalar(other):
            self.equalize_norms(divisor=other)
            return self
        else:
            return NotImplemented
        

    # ref: https://arxiv.org/abs/0711.3960
    #
    # [bde=0] get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    #
    # O(chi^6)
    def get_left_transfer_eigen_ver1(self, bde=0, memo=None):
        if memo is None: memo = {}
        inket_memo, inbra_memo, outket_memo, outbra_memo = {}, {}, {}, {}

        TF_L = self.get_ket_bond(bde).fuse_indices(self.get_ket_left_labels_bond(bde), fusedLabel=unique_label(), output_memo=inket_memo)
        TF_L *= self.get_bra_bond(bde).fuse_indices(self.get_bra_left_labels_bond(bde), fusedLabel=unique_label(), output_memo=inbra_memo)
        for i in range(bde, bde+len(self)-1):
            TF_L *= self.get_ket_site(i)
            TF_L *= self.get_bra_site(i)
            TF_L *= self.get_ket_bond(i+1)
            TF_L *= self.get_bra_bond(i+1)
        TF_L *= self.get_ket_site(bde-1).fuse_indices(self.get_ket_right_labels_site(bde-1), fusedLabel=unique_label(), output_memo=outket_memo)
        TF_L *= self.get_bra_site(bde-1).fuse_indices(self.get_bra_right_labels_site(bde-1), fusedLabel=unique_label(), output_memo=outbra_memo)

        W_L, V_L = tnd.tensor_eigsh(TF_L, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_L.hermite(inket_memo["fusedLabel"], inbra_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_L.split_index(input_memo=inket_memo, inplace=True)
        V_L.split_index(input_memo=inbra_memo, inplace=True)

        return W_L, V_L

    # ref: https://arxiv.org/abs/0711.3960
    #
    # [bde=0] get R s.t.
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        R  ==  c *  R
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    #
    # O(chi^6)
    def get_right_transfer_eigen_ver1(self, bde=0, memo=None):
        if memo is None: memo = {}
        inket_memo, inbra_memo, outket_memo, outbra_memo = {}, {}, {}, {}

        TF_R = self.get_ket_bond(bde).fuse_indices(self.get_ket_right_labels_bond(bde), fusedLabel=unique_label(), output_memo=inket_memo)
        TF_R *= self.get_bra_bond(bde).fuse_indices(self.get_bra_right_labels_bond(bde), fusedLabel=unique_label(), output_memo=inbra_memo)
        for i in range(bde+len(self)-1, bde, -1):
            TF_R *= self.get_ket_site(i)
            TF_R *= self.get_bra_site(i)
            TF_R *= self.get_ket_bond(i)
            TF_R *= self.get_bra_bond(i)
        TF_R *= self.get_ket_site(bde).fuse_indices(self.get_ket_left_labels_site(bde), fusedLabel=unique_label(), output_memo=outket_memo)
        TF_R *= self.get_bra_site(bde).fuse_indices(self.get_bra_left_labels_site(bde), fusedLabel=unique_label(), output_memo=outbra_memo)

        W_R, V_R = tnd.tensor_eigsh(TF_R, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_R.hermite(inbra_memo["fusedLabel"], inket_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_R.split_index(input_memo=inket_memo, inplace=True)
        V_R.split_index(input_memo=inbra_memo, inplace=True)
        
        return W_R, V_R



    # ref: https://arxiv.org/abs/1512.04938
    #
    # [bde=0] get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-            /-
    # L        |                 |      ==  W * L
    # \-(0)-[0]-...-(len-1)-[len-1]-            \-
    #
    # O(chi^3 * phys_dim * repeat)
    def get_left_transfer_eigen_ver2(self, bde=0, memo=None):
        if memo is None: memo = {}
        tshape = self.get_right_shape_site(bde-1)
        tlabels = self.get_ket_right_labels_site(bde-1)
        trdim = soujou(tshape)
        trlabel = unique_label()
        T = tni.random_tensor((trdim,)+tshape, [trlabel]+tlabels)
        V_L = T * T.adjoint(tlabels)

        for iteri in range(1,101):
            T.normalize(inplace=True)
            old_V_L = T * T.adjoint(tlabels)

            for e in range(bde,bde+len(self)):
                D = T*self.get_ket_bond(e)*self.get_ket_site(e)
                _,T = D.qr([trlabel]+self.get_phys_labels_site(e), self.get_ket_right_labels_site(e), qr_labels=trlabel)

            V_L = T * T.adjoint(tlabels)
            temp = V_L.is_prop_to(old_V_L, check_rtol=1e-14, check_atol=1e-14)
            W_L = np.real(temp["factor"])
            if temp:
                break
        memo["iter_times"] = iteri

        return W_L, V_L

    # ref: https://arxiv.org/abs/1512.04938
    #
    # [bde=0] get R s.t.
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        R  ==  W *  R
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    #
    # O(chi^3 * phys_dim * repeat)
    def get_right_transfer_eigen_ver2(self, bde=0,  memo=None):
        if memo is None: memo = {}
        tshape = self.get_left_shape_site(bde)
        tlabels = self.get_ket_left_labels_site(bde)
        trdim = soujou(tshape)
        trlabel = unique_label()
        T = tni.random_tensor((trdim,)+tshape, [trlabel]+tlabels)

        for iteri in range(1,101):
            T.normalize(inplace=True)
            old_V_R = T * T.adjoint(tlabels)

            for e in range(bde+len(self)-1, bde-1, -1):
                D = T*self.get_ket_bond(e+1)*self.get_ket_site(e)
                T,_ = D.lq(self.get_ket_left_labels_site(e), self.get_phys_labels_site(e)+[trlabel], lq_labels=trlabel)

            V_R = T * T.adjoint(tlabels)
            temp = V_R.is_prop_to(old_V_R, check_rtol=1e-14, check_atol=1e-14)
            W_R = np.real(temp["factor"])
            if temp:
                break
        memo["iter_times"] = iteri

        return W_R, V_R

    get_left_transfer_eigen = get_left_transfer_eigen_ver2
    get_right_transfer_eigen = get_right_transfer_eigen_ver2


    # ref: https://arxiv.org/abs/1512.04938
    #
    # [bde=0] get L s.t. (w will be the maximum one of availables)
    # /-L-(0)-[0]-...-(len-1)-[len-1]-                 /-L-
    # |        |                 |      == w * (unitary tensor)
    #
    # O(chi^3 * phys_dim * repeat)
    def get_left_half_transfer_eigen(self, bde=0, memo=None, edge_label=None):
        if memo is None: memo = {}
        tshape = self.get_right_shape_site(bde-1)
        tlabels = self.get_ket_right_labels_site(bde-1)
        trdim = soujou(tshape)
        trlabel = unique_label() if edge_label is None else edge_label
        T = tni.random_tensor((trdim,)+tshape, [trlabel]+tlabels)
        T = T / T.norm()

        for iteri in range(1,101):
            old_T = T

            for e in range(bde,bde+len(self)):
                D = T*self.get_ket_bond(e)*self.get_ket_site(e)
                _,T = D.qr([trlabel]+self.get_phys_labels_site(e), self.get_ket_right_labels_site(e), qr_labels=trlabel, force_diagonal_elements_positive=True)

            w = T.norm()
            T = T / w

            if T.__eq__(old_T, check_rtol=1e-14, check_atol=1e-14):
                break

        memo["iter_times"] = iteri

        return w,T

    # ref: https://arxiv.org/abs/1512.04938
    #
    # [bde=0] get R s.t. (w will be the maximum one of availables)
    # -[0]-...-(len-1)-[len-1]-(0)-R-\               -R-\
    #   |                 |          |  ==  w *  (unitary tensor)
    #
    # O(chi^3 * phys_dim * repeat)
    def get_right_half_transfer_eigen(self, bde=0, memo=None, edge_label=None):
        if memo is None: memo = {}
        tshape = self.get_left_shape_site(bde)
        tlabels = self.get_ket_left_labels_site(bde)
        trdim = soujou(tshape)
        trlabel = unique_label() if edge_label is None else edge_label
        T = tni.random_tensor((trdim,)+tshape, [trlabel]+tlabels)
        T = T / T.norm()

        for iteri in range(1,101):
            old_T = T

            for e in range(bde+len(self)-1, bde-1, -1):
                D = T*self.get_ket_bond(e+1)*self.get_ket_site(e)
                T,_ = D.lq(self.get_ket_left_labels_site(e), self.get_phys_labels_site(e)+[trlabel], lq_labels=trlabel, force_diagonal_elements_positive=True)

            w = T.norm()
            T = T / w

            if T.__eq__(old_T, check_rtol=1e-14, check_atol=1e-14):
                break

        memo["iter_times"] = iteri

        return w,T


    # ref: https://arxiv.org/abs/0711.3960
    #
    # [bde=0]
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # |      |                 |      ==  c * |
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    #
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        |  ==  c *  |
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    #
    # O(chi^6) + O(chi^6)
    def universally_canonize_around_end_bond_ver1(self, bde=0, chi=None, decomp_rtol=1e-20, decomp_atol=1e-30, transfer_normalize=True, memo=None):
        if memo is None: memo = {}
        dl_label = unique_label()
        dr_label = unique_label()
        memo["left_transfer_eigen"] = {}
        W_L, V_L = self.get_left_transfer_eigen(bde=bde, memo=memo["left_transfer_eigen"])
        memo["right_transfer_eigen"] = {}
        W_R, V_R = self.get_right_transfer_eigen(bde=bde, memo=memo["right_transfer_eigen"])
        W = sqrt(W_L*W_R)
        #assert abs(W_L-W_R) < 1e-3*abs(w), f"transfer_eigen different. {W_L} != {W_R}"
        memo["W_L"] = W_L
        memo["W_R"] = W_R
        memo["W"] = W
        w = sqrt(W)
        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_ket_left_labels_bond(bde), self.get_bra_left_labels_bond(bde), eigh_labels=dl_label)
        Y.unaster_labels(self.get_bra_left_labels_bond(bde), inplace=True)
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_ket_right_labels_bond(bde), self.get_bra_right_labels_bond(bde), eigh_labels=dr_label)
        Xh.unaster_labels(self.get_bra_right_labels_bond(bde), inplace=True)
        l0 = self.bdts[bde]
        G = d_L.sqrt() * Yh * l0 * X * d_R.sqrt()
        U, S, V = tnd.tensor_svd(G, dl_label, dr_label, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)
        M = Y * d_L.inv().sqrt() * U
        N = V * d_R.inv().sqrt() * Xh
        # l0 == M*S*N
        self.tensors[bde] = N * self.tensors[bde]
        self.tensors[bde-1] = self.tensors[bde-1] * M
        if transfer_normalize:
            self.bdts[bde] = S / w
            return w
        else:
            self.bdts[bde] = S
            return 1.0


    # ref: https://arxiv.org/abs/1512.04938
    # O(chi^3 * phys_dim * repeat * 2 + chi^3)
    def universally_canonize_around_end_bond(self, bde=0, chi=None, decomp_rtol=1e-20, decomp_atol=1e-30, transfer_normalize=True, memo=None):
        if memo is None: memo = {}
        dl_label = unique_label()
        dr_label = unique_label()
        memo["left_half_transfer_eigen"] = {}
        w_L, T_L = self.get_left_half_transfer_eigen(bde=bde, memo=memo["left_half_transfer_eigen"],edge_label=dl_label)
        memo["right_half_transfer_eigen"] = {}
        w_R, T_R = self.get_right_half_transfer_eigen(bde=bde, memo=memo["right_half_transfer_eigen"],edge_label=dr_label)
        #print("wlr",w_L, w_R)
        if w_L<0 or w_R<0:
            raise InternalError(f"In universally_canonize_around_end_bond, internal calculation conflicted: w_L={w_L}, w_R={w_R}) (both must be positive)")
        w = sqrt(w_L*w_R)
        #assert abs(W_L-W_R) < 1e-3*abs(w), f"transfer_eigen different. {W_L} != {W_R}"
        memo["w_L"] = w_L
        memo["w_R"] = w_R
        memo["w"] = w
        l0 = self.bdts[bde]
        U, S, V = tnd.tensor_svd(T_L*l0*T_R, dl_label, dr_label, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)

        M = l0 * T_R * V.conjugate() * S.inv() # == T_L.inv() * U
        N = S.inv() * U.conjugate() * T_L * l0 # == V * T_R.inv()

        # l0 == M*S*N
        self.tensors[bde-1] = self.tensors[bde-1] * M
        self.bdts[bde] = S
        self.tensors[bde] = N * self.tensors[bde]

        weight = w if transfer_normalize else 1.0

        self /= weight

        return weight


    locally_left_canonize_around_right_end = NotImplemented
    locally_right_canonize_around_left_end = NotImplemented

    def locally_left_canonize_around_bond(self, bde, chi=None, decomp_rtol=1e-20, decomp_atol=1e-30):
        return self.locally_left_canonize_around_not_end_bond(bde, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)

    def locally_right_canonize_around_bond(self, bde, chi=None, decomp_rtol=1e-20, decomp_atol=1e-30):
        return self.locally_right_canonize_around_not_end_bond(bde, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)


    def universally_canonize(self, left_already=0, right_already=None, chi=None, decomp_rtol=1e-20, decomp_atol=1e-30, transfer_normalize=True, memo=None):
        if memo is None: memo={}
        weight = 1.0

        if left_already == 0 and right_already is None:
            weight *= self.universally_canonize_around_end_bond(0, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol, transfer_normalize=transfer_normalize, memo=memo)
            for i in range(1, len(self)):
                weight *= self.locally_left_canonize_around_bond(i, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)
            for i in range(len(self)-1,0,-1):
                weight *= self.locally_right_canonize_around_bond(i, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)
        else:
            if right_already is None: right_already = len(self)
            weight *= self.globally_left_canonize_upto(right_already-1, left_already, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)
            weight *= self.globally_right_canonize_upto(left_already+1, right_already, chi=chi, decomp_rtol=decomp_rtol, decomp_atol=decomp_atol)

        return weight

    canonize = universally_canonize



    # applying methods
    def apply(self, gate, offset=0, chi=None, keep_universal_canonicality=True, keep_phys_labels=True):
        if type(gate)==Inf1DSImBTPO:
            tnop.apply_inf1DSimBTPS_inf1DSimBTPO(self,gate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
        elif type(gate) in [Opn1DBTPO, Opn1DTPO, Opn1DTMO]:
            gate = gate.to_BTPO()
            tnop.apply_inf1DSimBTPS_fin1DSimBTPO(self,gate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
        else: # list of gates
            for reallygate in gate:
                self.apply(reallygate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
            return

    def apply_everyplace(self, gate, chi=None, keep_universal_canonicality=True, gating_order="grissand"):
        if type(gate)==Inf1DSImBTPO:
            pass
        elif type(gate) in [Opn1DBTPO, Opn1DTPO, Opn1DTMO]:
            gate = gate.to_BTPO()
        else:
            for reallygate in gate:
                self.apply_everyplace(reallygate,chi=chi,keep_universal_canonicality=keep_universal_canonicality,gating_order=gating_order)
            return
        if gating_order == "grissand":
            for k in range(len(self)):
                self.apply(gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)
        elif gating_order == "trill":
            for i in range(len(gate)):
                for k in range(i,len(self),len(gate)):
                    self.apply(gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)

