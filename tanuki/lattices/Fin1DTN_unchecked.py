import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
import textwrap
from math import sqrt

#Finite 1D Simple Tensor Product State (wholly used when making BTPS through)
class Fin1DSimTPS:
    def __init__(self, tensors, phys_labelss=None, left_ext_bdt=None, right_ext_bdt=None):
        self.tensors = tensors
        if left_ext_bdt is None:
            self.left_ext_bdt = tni.dummy_diagonalTensor()
            self.tensors[0] = self.tensors[0].add_dummy_index()
        else: 
            self.left_ext_bdt = left_ext_bdt
        if right_ext_bdt is None:
            self.right_ext_bdt = tni.dummy_diagonalTensor()
            self.tensors[-1] = self.tensors[-1].add_dummy_index()
        else:
            self.right_ext_bdt = right_ext_bdt
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = phys_labelss

    def __copy__(self):
        return Fin1DSimTPS(copyModule.copy(self.tensors), phys_labelss=copyModule.copy(self.phys_labelss), left_ext_bdt=copyModule.copy(left_ext_bdt), right_ext_bdt=copyModule.copy(right_ext_bdt))

    def __deepcopy__(self, memo=None):
        if memo is None: memo = {}
        return Fin1DSimTPS(copyModule.deepcopy(self.tensors), phys_labelss=copyModule.deepcopy(self.phys_labelss), left_ext_bdt=copyModule.deepcopy(self.left_ext_bdt), right_ext_bdt=copyModule.deepcopy(self.right_ext_bdt))

    def copy(self, shallow=False):
        if shallow:
            return self.__copy__()
        else:
            return self.__deepcopy__()


    def __repr__(self):
        return f"Fin1DSimTPS(tensors={self.tensors}, phys_labelss={self.phys_labelss}, left_ext_bdt={self.left_ext_bdt}, right_ext_bdt={self.right_ext_bdt})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for tensor in self.tensors:
                dataStr += str(tensor)
                dataStr += ",\n"

        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr += f"left_ext_bdt={self.left_ext_bdt},\n"
        dataStr += f"right_ext_bdt={self.right_ext_bdt}\n"

        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"Fin1DSimTPS(\n" + \
        dataStr + \
        f")"

        return re


    def __len__(self):
        return self.tensors.__len__()

    def __iter__(self):
        return self.tensors.__iter__()

    def __getitem__(self, key):
        return self.tensors.__getitem__(key)

    def __setitem__(self, key, value):
        self.tensors.__setitem__(key, value)


    def get_labels_site(self, site):
        if site==-1:
            return self.left_ext_bdt.labels
        if site==len(self):
            return self.right_ext_bdt.labels
        return self.tensors[site].labels

    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.get_labels_site(site-1), self.get_labels_site(site))

    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.get_labels_site(site), self.get_labels_site(site+1))

    def get_phys_labels_site(self, site):
        return self.phys_labelss[site]

    def get_guessed_phys_labels_site(self, site):
        return diff_list(self.tensors[site].labels, self.get_left_labels_site(site)+self.get_right_labels_site(site))


    def get_left_dims_site(self, site):
        return self.tensors[site].dims(self.get_left_labels_site(site))

    def get_right_dims_site(self, site):
        return self.tensors[site].dims(self.get_right_labels_site(site))


    # (site=1):
    # /-[1]-      /-
    # |  |    ==  |
    # \-[1]-      \-
    def left_canonize_site(self, site, chi=None, rtol=None, atol=None):
        if site==len(self)-1:
            return
        U, S, V = tnd.truncated_svd(self[site], self.get_left_labels_site(site)+self.get_phys_labels_site(site), chi=chi, rtol=rtol, atol=atol)
        self[site] = U
        self[site+1] = S*V*self[site+1]

    # (site=4):
    # -[4]-\      -\
    #   |  |  ==   |
    # -[4]-/      -/
    def right_canonize_site(self, site, chi=None, rtol=None, atol=None):
        if site==0:
            return
        U, S, V = tnd.truncated_svd(self[site], self.get_left_labels_site(site), chi=chi, rtol=rtol, atol=atol)
        self[site] = V
        self[site-1] = self[site-1]*U*S

    # (interval=2):
    # /-[0]-      /-[1]-      /-
    # |  |    ==  |  |    ==  |
    # \-[0]-      \-[1]-      \-
    def left_canonize_upto(self, interval=None, chi=None, rtol=None, atol=None):
        if interval is None: interval = len(self)
        assert 0<=interval<=len(self)
        for site in range(interval):
            self.left_canonize_site(site, chi=chi, rtol=rtol, atol=atol)

    # (interval=4):
    # -[4]-\      -[5]-\      -\
    #   |  |  ==    |  |  ==   |
    # -[4]-/      -[5]-/      -/
    def right_canonize_upto(self, interval=0, chi=None, rtol=None, atol=None):
        assert 0<=interval<=len(self)
        for site in range(len(self)-1, interval-1, -1):
            self.right_canonize_site(site, chi=chi, rtol=rtol, atol=atol)

    def fuse_all_int_virt_indices(self):
        for site in range(len(self)-1):
            commons = self.get_right_labels_site(site)
            if len(commons) != 1:
                newLabel = tnc.unique_label()
                self.tensors[site].fuse_indices(commons, newLabel)
                self.tensors[site+1].fuse_indices(commons, newLabel)

    def to_BTPS(self):
        copied = copyModule.deepcopy(self)
        bdts = []
        bdts.append(copied.left_ext_bdt)
        for site in range(len(self)-1):
            halfshape = self.get_right_dims_site(site)
            halflabels = self.get_right_labels_site(site)
            bdt = tni.identity_diagonalTensor(halfshape, labels=halflabels)
            bdts.append(bdt)
        bdts.append(copied.right_ext_bdt)
        return Fin1DSimBTPS(copied.tensors, bdts, phys_labelss=copied.phys_labelss)

    def to_tensor(self):
        re = tnc.Tensor(1)
        for x in self.tensors:
            re *= x
        re.remove_dummy_indices()
        return re



















    def left_canonize_right_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        site = len(self)-1

        if end_dealing=="no":
            return

        # just normalize.
        # this method senses only when it can be written as:
        # /-(len-1)-[len-1]-         /-
        # |            |      == c * |
        # \-(len-1)-[len-1]-         \-
        elif end_dealing=="normalize":
            S = self.bdts[site]
            V = self.tensors[site]
            SV = S*V
            #VhS = SV.conjugate()
            #VhSSV = VhS[self.get_right_labels_site(site-1)+self.get_phys_labels_site(site)]*SV[self.get_right_labels_site(site-1)+self.get_phys_labels_site(site)]
            #is_senseful = SV.is_left_unitary()
            norm = SV.norm()
            self.tensors[site] = V / norm
            return

        # expel norm to ext_bdt.
        # so including ext_bdt, this method keeps the norm.
        # this method senses only when it can be written as:
        # /-(len-1)-[len-1]-       /-
        # |            |      ==  (*)
        # \-(len-1)-[len-1]-       \-
        elif end_dealing=="expel_bdt":
            return "e-n yappa dounimo DiagonalTensor motto jigen hoshii yo~" #TODO

        # svd including bdt[len] and expel the remain V as ext_ut and return.
        # so the followings are held:
        #     /-(len-1)-[len-1]-       /-
        # new |            |      ==   |
        #     \-(len-1)-[len-1]-       \-
        #
        #     /-(len-1)-[len-1]-(len)-           (len)-[ext_ut]-
        # old |            |            ==  new    |
        #     \-(len-1)-[len-1]-(len)-           (len)-[ext_ut]-
        # this method senses in all case.
        elif end_dealing=="expel_ut":
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site)+self.get_phys_labels_site(site), chi=chi, rtol=rtol, atol=atol)
            self.tensors[site] = U/self.bdts[site]
            self.bdts[site+1] = S
            return V

        else:
            return



    def right_canonize_left_end(self, chi=None, rtol=None, atol=None, end_dealing="normalize"):
        site = 0
        if end_dealing=="no":
            return
        elif end_dealing=="normalize":
            U = self.tensors[site]
            S = self.bdts[site+1]
            US = U*S
            norm = US.norm()
            self.tensors[site] = U / norm
            return
        elif end_dealing=="expel_bdt":
            return "e-n yappa dounimo DiagonalTensor motto jigen hoshii yo~" #TODO
        elif end_dealing=="expel_ut":
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site), chi=chi, rtol=rtol, atol=atol)
            self.bdts[site] = S
            self.tensors[site] = V/self.bdts[site+1]
            return U
        else:
            return















    # ref: https://arxiv.org/abs/0711.3960
    def canonize_end(self, chi=None, rtol=None, atol=None, normalize=True):
        dl_label = unique_label()
        dr_label = unique_label()
        w_L, V_L = self.get_left_transfer_eigen()
        w_R, V_R = self.get_right_transfer_eigen()
        assert abs(w_L-w_R) < 1e-10*abs(w_L)
        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_right_labels_site(len(self)-1), aster_labels(self.get_right_labels_site(len(self)-1)), eigh_labels=dl_label)
        Y.unaster_labels(aster_labels(self.get_right_labels_site(len(self)-1)))
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_left_labels_site(0), aster_labels(self.get_left_labels_site(0)), eigh_labels=dr_label)
        Xh.unaster_labels(aster_labels(self.get_left_labels_site(0)))
        l0 = self.bdts[0]
        G = d_L.sqrt() * Yh * l0 * X * d_R.sqrt()
        U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi, rtol=rtol, atol=atol)
        M = Y * d_L.inv().sqrt() * U
        N = V * d_R.inv().sqrt() * Xh
        # l0 == M*S*N
        if normalize:
            self.bdts[0] = S / sqrt(w_L)
        else:
            self.bdts[0] = S
        self.tensors[0] = N * self.tensors[0]
        self.tensors[len(self)-1] = self.tensors[len(self)-1] * M

    left_canonize_site = Fin1DSimBTPS.left_canonize_not_end_site

    def canonize(self, chi=None, rtol=None, atol=None):
        self.canonize_end(chi=chi, rtol=rtol, atol=atol)
        for i in range(len(self)-1):
            self.left_canonize_site(i, chi=chi, rtol=rtol, atol=atol)
        for i in range(len(self)-1,0,-1):
            self.right_canonize_site(i, chi=chi, rtol=rtol, atol=atol)

    def is_canonical(self):
        for i in range(len(self)):
            if not self.is_left_canonical_site(i):
                return False
        for i in range(len(self)-1,-1,-1):
            if not self.is_right_canonical_site(i):
                return False
        return True