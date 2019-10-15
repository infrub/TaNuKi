from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.OBTPS import Obc1DBTPS

# ██ ██████  ████████ ██████  ███████ 
# ██ ██   ██    ██    ██   ██ ██      
# ██ ██████     ██    ██████  ███████ 
# ██ ██   ██    ██    ██           ██ 
# ██ ██████     ██    ██      ███████ 
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DBTPS(MixinInf1DBTP_, Obc1DBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        self.bdts = CyclicList(bdts)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Inf1DBTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += str(bdt)
                dataStr += "\n"
                tensor = self.tensors[i]
                dataStr += str(tensor)
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



    # [bde=0] get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    #
    # O(chi^6)
    def get_left_transfer_eigen(self, bde=0):
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

        w_L, V_L = tnd.tensor_eigsh(TF_L, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_L.hermite(inket_memo["fusedLabel"], inbra_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_L.split_index(input_memo=inket_memo, inplace=True)
        V_L.split_index(input_memo=inbra_memo, inplace=True)

        return w_L, V_L

    # [bde=0] get R s.t.
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        R  ==  c *  R
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    #
    # O(chi^6)
    def get_right_transfer_eigen(self, bde=0):
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

        w_R, V_R = tnd.tensor_eigsh(TF_R, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_R.hermite(inbra_memo["fusedLabel"], inket_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_R.split_index(input_memo=inket_memo, inplace=True)
        V_R.split_index(input_memo=inbra_memo, inplace=True)
        
        return w_R, V_R



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
    def universally_canonize_around_end_bond(self, bde=0, chi=None, rtol=None, atol=None, transfer_normalize=True):
        dl_label = unique_label()
        dr_label = unique_label()
        w_L, V_L = self.get_left_transfer_eigen(bde=bde)
        w_R, V_R = self.get_right_transfer_eigen(bde=bde)
        assert abs(w_L-w_R) < 1e-10*abs(w_L)
        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_ket_left_labels_bond(bde), self.get_bra_left_labels_bond(bde), eigh_labels=dl_label)
        Y.unaster_labels(self.get_bra_left_labels_bond(bde), inplace=True)
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_ket_right_labels_bond(bde), self.get_bra_right_labels_bond(bde), eigh_labels=dr_label)
        Xh.unaster_labels(self.get_bra_right_labels_bond(bde), inplace=True)
        l0 = self.bdts[bde]
        G = d_L.sqrt() * Yh * l0 * X * d_R.sqrt()
        U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi, rtol=rtol, atol=atol)
        M = Y * d_L.inv().sqrt() * U
        N = V * d_R.inv().sqrt() * Xh
        # l0 == M*S*N
        if transfer_normalize:
            self.bdts[bde] = S / sqrt(w_L)
        else:
            self.bdts[bde] = S
        self.tensors[bde] = N * self.tensors[bde]
        self.tensors[bde-1] = self.tensors[bde-1] * M

    locally_left_canonize_around_right_end = NotImplemented
    locally_right_canonize_around_left_end = NotImplemented

    def locally_left_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None):
        self.locally_left_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)

    def locally_right_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None):
        self.locally_right_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)


    def universally_canonize(self, left_already=0, right_already=None, chi=None, rtol=None, atol=None, transfer_normalize=True):
        if left_already == 0 and right_already is None:
            self.universally_canonize_around_end_bond(0, chi=chi, rtol=rtol, atol=atol, transfer_normalize=transfer_normalize)
            for i in range(1, len(self)):
                self.locally_left_canonize_around_bond(i, chi=chi, rtol=rtol, atol=atol)
            for i in range(len(self)-1,0,-1):
                self.locally_right_canonize_around_bond(i, chi=chi, rtol=rtol, atol=atol)
            """
            self.globally_left_canonize_upto(len(self)-1, 0, chi=chi, rtol=rtol, atol=atol, transfer_normalize=transfer_normalize)
            self.globally_right_canonize_upto(1, len(self), chi=chi, rtol=rtol, atol=atol, transfer_normalize=transfer_normalize)
            """
        else:
            if right_already is None: right_already = len(self)
            self.globally_left_canonize_upto(right_already-1, left_already, chi=chi, rtol=rtol, atol=atol)
            self.globally_right_canonize_upto(left_already+1, right_already, chi=chi, rtol=rtol, atol=atol)

    canonize = universally_canonize



    # applying methods
    def apply(self, gate, offset=0, chi=None, keep_universal_canonicality=True, keep_phys_labels=True):
        if type(gate)==Inf1DSImBTPO:
            tnop.apply_inf1DSimBTPS_inf1DSimBTPO(self,gate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
        elif type(gate) in [Obc1DBTPO, Obc1DTPO, Obc1DTMO]:
            gate = gate.to_BTPO()
            tnop.apply_inf1DSimBTPS_fin1DSimBTPO(self,gate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
        else: # list of gates
            for reallygate in gate:
                self.apply(reallygate,offset=offset,chi=chi,keep_universal_canonicality=keep_universal_canonicality,keep_phys_labels=keep_phys_labels)
            return

    def apply_everyplace(self, gate, chi=None, keep_universal_canonicality=True, gating_order="grissand"):
        if type(gate)==Inf1DSImBTPO:
            pass
        elif type(gate) in [Obc1DBTPO, Obc1DTPO, Obc1DTMO]:
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

