import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
import textwrap
from math import sqrt



# bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- tensors[len-1] -- bdts[len]
class Fin1DSimBTPS:
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


    # getting label methods (to be not inherited)
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

    # getting label methods (to be inherited)
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



    # canonizing methods

    # (site=1):
    # /-(1)-[1]-      /-
    # |      |    ==  |
    # \-(1)-[1]-      \-
    def left_canonize_not_end_site(self, site, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site)+self.get_phys_labels_site(site), chi=chi, rtol=rtol, atol=atol)
        self.tensors[site] = U/self.bdts[site]
        self.bdts[site+1] = S
        self.tensors[site+1] = V*self.tensors[site+1]

    def left_canonize_right_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        site = len(self)-1
        if self.bdts[site+1].size==1:
            if end_dealing=="no":
                pass
            elif end_dealing=="normalize":
                self.tensors[site] /= (self.bdts[site]*self.tensors[site]).norm()
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                pass
            else:
                raise UndecidedError

    def left_canonize_site(self, site, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if site==len(self)-1:
            self.left_canonize_right_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        else:
            self.left_canonize_not_end_site(site, chi=chi, rtol=rtol, atol=atol)

    # (site=4):
    # -[4]-(5)-\      -\
    #   |      |  ==   |
    # -[4]-(5)-/      -/
    def right_canonize_not_end_site(self, site, chi=None, rtol=None, atol=None):
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site), chi=chi, rtol=rtol, atol=atol)
        self.tensors[site-1] = self.tensors[site-1]*U
        self.bdts[site] = S
        self.tensors[site] = V/self.bdts[site+1]

    def right_canonize_left_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        site = 0
        if self.bdts[site].size==1:
            if end_dealing=="no":
                pass
            elif end_dealing=="normalize":
                self.tensors[site] /= (self.tensors[site]*self.bdts[site+1]).norm()
            else:
                raise UndecidedError
        else:
            if end_dealing=="no":
                pass
            else:
                raise UndecidedError

    def right_canonize_site(self, site, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        if site==0:
            self.right_canonize_left_end(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        else:
            self.right_canonize_not_end_site(site, chi=chi, rtol=rtol, atol=atol)



    def normarg_slice(self, slice_start=None, slice_end=None):
        if slice_start is None: slice_start = 0
        if slice_end is None: slice_end = len(self)
        slice_start, slice_end = min(slice_start, slice_end), max(slice_start, slice_end)
        if 0>slice_start or slice_end>len(self):
            raise ValueError
        return slice_start, slice_end

    # (0:2):
    # /-(0)-[0]-      /-(1)-[1]-      /-
    # |      |    ==  |      |    ==  |
    # \-(0)-[0]-      \-(1)-[1]-      \-
    def left_canonize(self, slice_start=None, slice_end=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        slice_start, slice_end = self.normarg_slice(slice_start, slice_end)
        for site in range(slice_start, slice_end):
            self.left_canonize_site(site, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    # (4:6):
    # -[4]-(5)-\      -[5]-(6)-\      -\
    #   |      |  ==    |      |  ==   |
    # -[4]-(5)-/      -[5]-(6)-/      -/
    def right_canonize(self, slice_start=None, slice_end=None, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        slice_start, slice_end = self.normarg_slice(slice_start, slice_end)
        for site in range(slice_end-1, slice_start-1, -1):
            self.right_canonize_site(site, chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)

    def both_canonize(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        self.left_canonize(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)
        self.right_canonize(chi=chi, rtol=rtol, atol=atol, end_dealing=end_dealing)



    def is_left_canonical_site(self, site):
        S = self.bdts[site]
        V = self.tensors[site]
        SV = S*V
        re = SV.is_prop_semi_unitary(self.get_left_labels_bond(site)+self.get_phys_labels_site(site))
        #print(site,re)
        return re

    def is_right_canonical_site(self, site):
        U = self.tensors[site]
        S = self.bdts[site+1]
        US = U*S
        re = US.is_prop_semi_unitary(self.get_phys_labels_site(site)+self.get_right_labels_bond(site+1))
        #print(site,re)
        return re

    def is_left_canonical(self, slice_start=None, slice_end=None):
        slice_start, slice_end = self.normarg_slice(slice_start, slice_end)
        ok = True
        res = {}
        for site in range(slice_start, slice_end):
            re = self.is_left_canonical_site(site)
            res[site] = re
            ok = re and ok
        return CollateralBool(ok, res)

    def is_right_canonical(self, slice_start=None, slice_end=None):
        slice_start, slice_end = self.normarg_slice(slice_start, slice_end)
        ok = True
        res = {}
        for site in range(slice_end-1, slice_start-1, -1):
            re = self.is_right_canonical_site(site)
            res[site] = re
            ok = re and ok
        return CollateralBool(ok, res)

    def is_both_canonical(self, end_dealing="no"):
        left = self.is_left_canonical()
        right = self.is_right_canonical()
        return CollateralBool(left.trueOrFalse and right.trueOrFalse, {"left":left.expression, "right":right.expression})



    # converting methods
    def to_tensor(self):
        re = copyModule.deepcopy(self.bdts[0])
        for i in range(len(self)):
            re *= self.tensors[i]
            re *= self.bdts[i+1]
        return re





# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DSimBTPS(Fin1DSimBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        if type(tensors) != CyclicList:
            tensors = CyclicList(tensors)
        if type(bdts) != CyclicList:
            bdts = CyclicList(bdts)
        Fin1DSimBTPS.__init__(self, tensors, bdts, phys_labelss=phys_labelss)
        self.phys_labelss = CyclicList(self.phys_labelss)

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


    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)

    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site+1].labels)

    def get_left_labels_bond(self, bondsite):
        return self.get_right_labels_site(bondsite-1)

    def get_right_labels_bond(self, bondsite):
        return self.get_left_labels_site(bondsite)

    def to_tensor(self):
        re = 1
        for i in range(len(self)):
            re *= self.bdts[i]
            re *= self.tensors[i]
        return re



    # get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    def get_left_transfer_eigen(self):
        inket_memo, inbra_memo, outket_memo, outbra_memo = {}, {}, {}, {}

        TF_L = self.get_ket_bond(0).fuse_indices(self.get_ket_left_labels_bond(0), fusedLabel=unique_label(), output_memo=inket_memo)
        TF_L *= self.get_bra_bond(0).fuse_indices(self.get_bra_left_labels_bond(0), fusedLabel=unique_label(), output_memo=inbra_memo)
        for i in range(len(self)-1):
            TF_L *= self.get_ket_site(i)
            TF_L *= self.get_bra_site(i)
            TF_L *= self.get_ket_bond(i+1)
            TF_L *= self.get_bra_bond(i+1)
        TF_L *= self.get_ket_site(-1).fuse_indices(self.get_ket_right_labels_site(-1), fusedLabel=unique_label(), output_memo=outket_memo)
        TF_L *= self.get_bra_site(-1).fuse_indices(self.get_bra_right_labels_site(-1), fusedLabel=unique_label(), output_memo=outbra_memo)

        w_L, V_L = tnd.tensor_eigsh(TF_L, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_L.hermite(inket_memo["fusedLabel"], inbra_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_L.split_index(input_memo=inket_memo, inplace=True)
        V_L.split_index(input_memo=inbra_memo, inplace=True)

        return w_L, V_L

    # get R s.t.
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        R  ==  c *  R
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    def get_right_transfer_eigen(self):
        inket_memo, inbra_memo, outket_memo, outbra_memo = {}, {}, {}, {}

        TF_R = self.get_ket_bond(0).fuse_indices(self.get_ket_right_labels_bond(0), fusedLabel=unique_label(), output_memo=inket_memo)
        TF_R *= self.get_bra_bond(0).fuse_indices(self.get_bra_right_labels_bond(0), fusedLabel=unique_label(), output_memo=inbra_memo)
        for i in range(len(self)-1, 0, -1):
            TF_R *= self.get_ket_site(i)
            TF_R *= self.get_bra_site(i)
            TF_R *= self.get_ket_bond(i)
            TF_R *= self.get_bra_bond(i)
        TF_R *= self.get_ket_site(0).fuse_indices(self.get_ket_left_labels_site(0), fusedLabel=unique_label(), output_memo=outket_memo)
        TF_R *= self.get_bra_site(0).fuse_indices(self.get_bra_left_labels_site(0), fusedLabel=unique_label(), output_memo=outbra_memo)

        w_R, V_R = tnd.tensor_eigsh(TF_R, [outket_memo["fusedLabel"], outbra_memo["fusedLabel"]], [inket_memo["fusedLabel"], inbra_memo["fusedLabel"]])

        V_R.hermite(inbra_memo["fusedLabel"], inket_memo["fusedLabel"], assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_R.split_index(input_memo=inket_memo, inplace=True)
        V_R.split_index(input_memo=inbra_memo, inplace=True)
        
        return w_R, V_R











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

