import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
import textwrap
from math import sqrt





class Fin1DSimTPMixin:
    def get_left_labels_site(self, site):
        if site==0:
            return []
        return tnc.intersection_list(self.tensors[site-1].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        if site==len(self)-1:
            return []
        return tnc.intersection_list(self.tensors[site].labels, self.tensors[site+1].labels)

class Fin1DSimBTPMixin:
    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site+1].labels)
    def get_left_labels_bond(self, bondsite):
        if bondsite == 0:
            return diff_list(self.bdts[bondsite].labels, self.get_left_labels_site(bondsite))
        return self.get_right_labels_site(bondsite-1)
    def get_right_labels_bond(self, bondsite):
        if bondsite == len(self):
            return diff_list(self.bdts[bondsite].labels, self.get_right_labels_site(bondsite-1))
        return self.get_left_labels_site(bondsite)

class Inf1DSimTPMixin:
    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site-1].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.tensors[site+1].labels)

class Inf1DSimBTPMixin:
    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site+1].labels)
    def get_left_labels_bond(self, bondsite):
        return self.get_right_labels_site(bondsite-1)
    def get_right_labels_bond(self, bondsite):
        return self.get_left_labels_site(bondsite)



class _1DSimBTPSMixin:
    def get_phys_labels_site(self, site):
        return self.phys_labelss[site]

    def get_guessed_phys_labels_site(self, site):
        return diff_list(self.tensors[site].labels, self.get_left_labels_site(site)+self.get_right_labels_site(site))

    def get_left_shape_site(self, site):
        return self.tensors[site].dims(self.get_left_labels_site(site))

    def get_right_shape_site(self, site):
        return self.tensors[site].dims(self.get_right_labels_site(site))

    def get_ket_site(self, site):
        return self.tensors[site].copy(shallow=True)

    def get_bra_site(self, site):
        return self.tensors[site].adjoint(self.get_left_labels_site(site), self.get_right_labels_site(site), style="aster")

    def get_ket_bond(self, bondsite):
        return self.bdts[bondsite].copy(shallow=True)

    def get_bra_bond(self, bondsite):
        return self.bdts[bondsite].adjoint(self.get_left_labels_bond(bondsite), self.get_right_labels_bond(bondsite), style="aster")

    def get_ket_left_labels_site(self, site):
        return self.get_left_labels_site(site)

    def get_ket_right_labels_site(self, site):
        return self.get_right_labels_site(site)

    def get_bra_left_labels_site(self, site):
        return aster_labels(self.get_left_labels_site(site))

    def get_bra_right_labels_site(self, site):
        return aster_labels(self.get_right_labels_site(site))

    def get_ket_left_labels_bond(self, bondsite):
        return self.get_left_labels_bond(bondsite)

    def get_ket_right_labels_bond(self, bondsite):
        return self.get_right_labels_bond(bondsite)

    def get_bra_left_labels_bond(self, bondsite):
        return aster_labels(self.get_left_labels_bond(bondsite))

    def get_bra_right_labels_bond(self, bondsite):
        return aster_labels(self.get_right_labels_bond(bondsite))






# bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- tensors[len-1] -- bdts[len]
class Fin1DSimBTPS(Fin1DSimBTPMixin, _1DSimBTPSMixin):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = tensors
        self.bdts = bdts
        if len(self.bdts)+1==len(self.tensors):
            self.bdts = [dummy_diagonalTensor()] + self.bdts + [dummy_diagonalTensor()]
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = phys_labelss

    def __repr__(self):
        return f"Fin1DSimTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

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
            bdt = self.bdts[len(self)]
            dataStr += str(bdt)
            dataStr += "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"Fin1DSimBTPS(\n" + \
        dataStr + \
        f")"

        return re

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self)+1)]
        return Fin1DSimBTPS(tensors, bdts, self.phys_labelss)

    def to_tensor(self):
        re = self.bdts[0].copy(shallow=False)
        for i in range(len(self)):
            re *= self.tensors[i]
            re *= self.bdts[i+1]
        return re



    # canonizing methods
    
    # [Fin] bde = 1,2,..,len(self)-1
    # [bde=2]:
    # /-(1)-[1]-(2)-      /-(2)-
    # |      |        ==  |
    # \-(1)-[1]-(2)-      \-(2)-
    def locally_left_canonize_around_not_end_bond(self, bde, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[bde-1]*self.tensors[bde-1]*self.bdts[bde], self.get_left_labels_bond(bde-1)+self.get_phys_labels_site(bde-1), chi=chi, rtol=rtol, atol=atol)
        self.tensors[bde-1] = U/self.bdts[bde-1]
        self.bdts[bde] = S
        self.tensors[bde] = V*self.tensors[bde]

    # [Fin] bde = 1,..,len(self)-1
    # [bde=3]:
    # -(3)-[3]-(4)-\      -(3)-\
    #       |      |  ==       |
    # -(3)-[3]-(4)-/      -(3)-/
    def locally_right_canonize_around_not_end_bond(self, bde, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[bde]*self.tensors[bde]*self.bdts[bde+1], self.get_left_labels_bond(bde), chi=chi, rtol=rtol, atol=atol)
        self.tensors[bde-1] = self.tensors[bde-1]*U
        self.bdts[bde] = S
        self.tensors[bde] = V/self.bdts[bde+1]


    def locally_left_canonize_around_right_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        bde = len(self)
        if self.bdts[bde].size==1:
            if end_dealing=="no":
                pass
            elif end_dealing=="normalize":
                self.tensors[bde-1] /= (self.bdts[bde-1]*self.tensors[bde-1]).norm()
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                pass
            else:
                raise UndecidedError

    def locally_right_canonize_around_left_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        bde = 0
        if self.bdts[bde].size==1:
            if end_dealing=="no":
                pass
            elif end_dealing=="normalize":
                self.tensors[bde] /= (self.tensors[bde]*self.bdts[bde+1]).norm()
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                pass
            else:
                raise UndecidedError


    def locally_left_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if bde==0:
            pass
        elif bde==len(self):
            self.locally_left_canonize_around_right_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        else:
            self.locally_left_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)

    def locally_right_canonize_around_bond(self, bde, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if bde==0:
            self.locally_right_canonize_around_left_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        elif bde==len(self):
            pass
        else:
            self.locally_right_canonize_around_not_end_bond(bde, chi=chi, rtol=rtol, atol=atol)


    # [bde=2, already=0]
    # /-(0)-[0]-      /-(1)-[1]-      /-
    # |      |    ==  |      |    ==  |
    # \-(0)-[0]-      \-(1)-[1]-      \-
    def globally_left_canonize_upto(self, upto_bde=None, already_bde=0, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if upto_bde is None: upto_bde = len(self)
        for bde in range(already_bde+1, upto_bde+1):
            self.locally_left_canonize_around_bond(bde, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    # [bde=4, already=6]
    # -[4]-(5)-\      -[5]-(6)-\      -\
    #   |      |  ==    |      |  ==   |
    # -[4]-(5)-/      -[5]-(6)-/      -/
    def globally_right_canonize_upto(self, upto_bde=0, already_bde=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if already_bde is None: already_bde = len(self)
        for bde in range(already_bde-1, upto_bde-1, -1):
            self.locally_right_canonize_around_bond(bde, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    def universally_canonize(self, left_already=0, right_already=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if right_already is None: right_already = len(self)
        self.globally_left_canonize_upto(right_already, left_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        self.globally_right_canonize_upto(left_already, right_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    canonize = universally_canonize



    def is_locally_left_canonical_around_bond(self, bde):
        if bde == 0:
            return CollateralBool(True, {"factor":1.0})
        S = self.bdts[bde-1]
        V = self.tensors[bde-1]
        SV = S*V
        re = SV.is_prop_semi_unitary(self.get_left_labels_bond(bde-1)+self.get_phys_labels_site(bde-1))
        return re

    def is_locally_right_canonical_around_bond(self, bde):
        if bde == len(self):
            return CollateralBool(True, {"factor":1.0})
        U = self.tensors[bde]
        S = self.bdts[bde+1]
        US = U*S
        re = US.is_prop_semi_unitary(self.get_phys_labels_site(bde)+self.get_right_labels_bond(bde+1))
        return re

    def is_grobally_left_canonical_upto_bond(self, upto_bde=None):
        if upto_bde is None: upto_bde = len(self)
        ok = True
        res = {}
        for bde in range(0, upto_bde+1):
            re = self.is_locally_left_canonical_around_bond(bde)
            res[bde] = re
            if not re: ok = False
        return CollateralBool(ok, res)

    def is_grobally_right_canonical_upto_bond(self, upto_bde=0):
        ok = True
        res = {}
        for bde in range(len(self), upto_bde-1, -1):
            re = self.is_locally_right_canonical_around_bond(bde)
            res[bde] = re
            if not re: ok = False
        return CollateralBool(ok, res)

    def is_globally_both_canonical_upto_bond(self, upto_bde):
        return self.is_grobally_left_canonical_upto_bond(upto_bde) & self.is_grobally_right_canonical_upto_bond(upto_bde)

    def is_universally_canonical(self, bde_start=None, bde_end=None):
        return self.is_grobally_left_canonical_upto_bond() & self.is_grobally_right_canonical_upto_bond()

    is_canonical = is_universally_canonical










# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DSimBTPS(Inf1DSimBTPMixin, Fin1DSimBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        if type(tensors) != CyclicList:
            tensors = CyclicList(tensors)
        if type(bdts) != CyclicList:
            bdts = CyclicList(bdts)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        if type(phys_labelss) != CyclicList:
            phys_labelss = CyclicList(phys_labelss)
        self.tensors = tensors
        self.bdts = bdts
        self.phys_labelss = phys_labelss

    def __repr__(self):
        return f"Inf1DSimTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

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

        re = \
        f"Inf1DSimBTPS(\n" + \
        dataStr + \
        f")"

        return re

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self))]
        return Inf1DSimBTPS(tensors, bdts, self.phys_labelss)

    def to_tensor(self):
        re = 1
        for i in range(len(self)):
            re *= self.bdts[i]
            re *= self.tensors[i]
        return re



    # [bde=0] get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
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
    def universally_canonize_around_end_bond(self, bde=0, chi=None, rtol=None, atol=None, transfer_normalize=True):
        dl_label = unique_label()
        dr_label = unique_label()
        w_L, V_L = self.get_left_transfer_eigen(bde=bde)
        w_R, V_R = self.get_right_transfer_eigen(bde=bde)
        assert abs(w_L-w_R) < 1e-10*abs(w_L)
        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_ket_left_labels_bond(bde), self.get_bra_left_labels_bond(bde), eigh_labels=dl_label)
        Y.unaster_labels(self.get_bra_left_labels_bond(bde), inplace=True)
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_ket_right_labels_bond(bde), self.get_bra_right_labels_bond(bde), eigh_labels=dr_label)
        Xh.unaster_labels(self.get_bra_right_labels_bond(bde))
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
            self.globally_left_canonize_upto(len(self)-1, 0, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
            self.globally_right_canonize_upto(1, len(self), chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
            """
        else:
            if right_already is None: right_already = len(self)
            self.globally_left_canonize_upto(right_already, left_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
            self.globally_right_canonize_upto(left_already, right_already, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    canonize = universally_canonize





"""
class Fin1DSimTPO(Fin1DSimTPMixin):
    def __init__(self, tensors, physin_labelss, physout_labelss, is_unitary=False):
        self.tensors = tensors
        self.physin_labelss = physin_labelss
        self.physout_labelss = physout_labelss
        self.is_unitary = is_unitary

    def to_BTPO(self):
        pass



class Fin1DSimBTPO(Fin1DSimBTPMixin):
    def __init__(self, tensors, bdts, physin_labelss, physout_labelss, is_unitary=False):
        self.tensors = tensors
        self.bdts = bdts
        self.physin_labelss = physin_labelss
        self.physout_labelss = physout_labelss
        self.is_unitary = is_unitary








def apply_fin1DSimBTPS_fin1DSimTPO(self, mps, mpo):
    pass

"""
















def random_fin1DSimBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is not None and len(virt_labelss) == length-1:
        virt_labelss = [[]] + virt_labelss + [[]]
    if virt_dimss is not None and len(virt_dimss) == length-1:
        virt_dimss = [()] + virt_dimss + [()]

    if virt_labelss is None and virt_dimss is None:
        virt_labelss = [[]] + [[unique_label()] for _ in range(length-1)] + [[]]
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_dimss is None:
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_labelss is None:
        virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]

    bdts = []
    for bondsite in range(length+1):
        if len(virt_dimss[bondsite])==0:
            bdts.append( tni.dummy_diagonalTensor() )
        else:
            bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[site+1], virt_labelss[site]+phys_labelss[site]+virt_labelss[site+1] , dtype=dtype) )

    return Fin1DSimBTPS(tensors, bdts, phys_labelss)



def random_inf1DSimBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None and virt_dimss is None:
        virt_labelss = [[unique_label()] for _ in range(length)]
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_dimss is None:
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_labelss is None:
        virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]

    bdts = []
    for bondsite in range(length):
        if len(virt_dimss[bondsite])==0:
            bdts.append( tni.dummy_diagonalTensor() )
        else:
            bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Inf1DSimBTPS(tensors, bdts, phys_labelss)

