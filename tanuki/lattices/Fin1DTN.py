import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from numpy import prod as soujou
import textwrap

#Finite 1D Simple Tensor Product State (wholly used when making BTPS through)
class Fin1DSimTPS:
    def __init__(self, tensors, phys_labelss=None, left_ext_bdt=None, right_ext_bdt=None):
        self.tensors = tensors
        if left_ext_bdt is None:
            self.left_ext_bdt = tni.dummy_diagonalTensor()
            self.tensors[0].add_dummy_index()
        else: 
            self.left_ext_bdt = left_ext_bdt
        if right_ext_bdt is None:
            self.right_ext_bdt = tni.dummy_diagonalTensor()
            self.tensors[-1].add_dummy_index()
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


    def get_left_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_left_labels_site(site)))

    def get_right_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_right_labels_site(site)))


    # (site=1):
    # /-[1]-      /-
    # |  |    ==  |
    # \-[1]-      \-
    def left_canonize_site(self, site, chi=None, relative_threshold=1e-14):
        if site==len(self)-1:
            return
        U, S, V = tnd.truncated_svd(self[site], self.get_left_labels_site(site)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
        self[site] = U
        self[site+1] = S*V*self[site+1]

    # (site=4):
    # -[4]-\      -\
    #   |  |  ==   |
    # -[4]-/      -/
    def right_canonize_site(self, site, chi=None, relative_threshold=1e-14):
        if site==0:
            return
        U, S, V = tnd.truncated_svd(self[site], self.get_left_labels_site(site), chi=chi, relative_threshold=relative_threshold)
        self[site] = V
        self[site-1] = self[site-1]*U*S

    # (interval=2):
    # /-[0]-      /-[1]-      /-
    # |  |    ==  |  |    ==  |
    # \-[0]-      \-[1]-      \-
    def left_canonize_upto(self, interval=None, chi=None, relative_threshold=1e-14):
        if interval is None: interval = len(self)
        assert 0<=interval<=len(self)
        for site in range(interval):
            self.left_canonize_site(site, chi=chi, relative_threshold=relative_threshold)

    # (interval=4):
    # -[4]-\      -[5]-\      -\
    #   |  |  ==    |  |  ==   |
    # -[4]-/      -[5]-/      -/
    def right_canonize_upto(self, interval=0, chi=None, relative_threshold=1e-14):
        assert 0<=interval<=len(self)
        for site in range(len(self)-1, interval-1, -1):
            self.right_canonize_site(site, chi=chi, relative_threshold=relative_threshold)

    def fuse_all_int_virt_indices(self):
        for site in range(len(self)-1):
            commons = self.get_right_labels_site(site)
            if len(commons) != 1:
                newLabel = tnc.unique_label()
                self.tensors[site].fuse_indices(commons, newLabel)
                self.tensors[site+1].fuse_indices(commons, newLabel)

    def to_BTPS(self):
        fused_self = copyModule.deepcopy(self)
        fused_self.fuse_all_int_virt_indices()
        bdts = []
        bdts.append(fused_self.left_ext_bdt)
        for site in range(len(self)-1):
            dim = self.get_right_dim_site(site)
            label = self.get_right_labels_site(site)[0]
            bdt = tni.identity_tensor(dim, label)
            bdts.append(bdt)
        bdts.append(fused_self.right_ext_bdt)
        return Fin1DSimBTPS(fused_self.tensors, bdts, phys_labelss=fused_self.phys_labelss)

    def to_tensor(self):
        re = tnc.Tensor(1)
        for x in self.tensors:
            re *= x
        re.remove_all_dummy_indices()
        return re





# bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- tensors[len-1] -- bdts[len]
class Fin1DSimBTPS:
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = tensors
        self.bdts = bdts
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


    def get_left_labels_site(self, site):
        if site == len(self):
            return diff_list(self.bdts[site].labels, self.get_right_labels_site(site-1))
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)

    def get_right_labels_site(self, site):
        if site == -1:
            return diff_list(self.bdts[site+1].labels, self.get_left_labels_site(site+1))
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site+1].labels)

    def get_phys_labels_site(self, site):
        return self.phys_labelss[site]

    def get_guessed_phys_labels_site(self, site):
        return diff_list(self.tensors[site].labels, self.get_left_labels_site(site)+self.get_right_labels_site(site))

    def get_left_labels_bond(self, bondsite):
        if bondsite == 0:
            return diff_list(self.bdts[bondsite].labels, self.get_left_labels_site(bondsite))
        return self.get_right_labels_site(bondsite-1)

    def get_right_labels_bond(self, bondsite):
        if bondsite == len(self):
            return diff_list(self.bdts[bondsite].labels, self.get_right_labels_site(bondsite-1))
        return self.get_left_labels_site(bondsite)


    def get_left_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_left_labels_site(site)))

    def get_right_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_right_labels_site(site)))


    def left_canonize_right_end(self, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
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
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_right_labels_site(site-1)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
            self.tensors[site] = U/self.bdts[site]
            self.bdts[site+1] = S
            return V

        else:
            return

    # (site=1):
    # /-(1)-[1]-      /-
    # |      |    ==  |
    # \-(1)-[1]-      \-
    def left_canonize_site(self, site, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        if site==len(self)-1:
            self.left_canonize_right_end(chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_right_labels_site(site-1)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
        self.tensors[site] = U/self.bdts[site]
        self.bdts[site+1] = S
        self.tensors[site+1] = V*self.tensors[site+1]

    def right_canonize_left_end(self, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
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
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_right_labels_site(site-1), chi=chi, relative_threshold=relative_threshold)
            self.bdts[site] = S
            self.tensors[site] = V/self.bdts[site+1]
            return U
        else:
            return

    # (site=4):
    # -[4]-(5)-\      -\
    #   |      |  ==   |
    # -[4]-(5)-/      -/
    def right_canonize_site(self, site, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        if site==0:
            self.right_canonize_left_end(chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_right_labels_site(site-1), chi=chi, relative_threshold=relative_threshold)
        self.tensors[site-1] = self.tensors[site-1]*U
        self.bdts[site] = S
        self.tensors[site] = V/self.bdts[site+1]

    # (interval=2):
    # /-(0)-[0]-      /-(1)-[1]-      /-
    # |      |    ==  |      |    ==  |
    # \-(0)-[0]-      \-(1)-[1]-      \-
    def left_canonize_upto(self, interval=None, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        if interval is None: interval = len(self)
        assert 0<=interval<=len(self)
        for site in range(interval):
            self.left_canonize_site(site, chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)

    # (interval=4):
    # -[4]-(5)-\      -[5]-(6)-\      -\
    #   |      |  ==    |      |  ==   |
    # -[4]-(5)-/      -[5]-(6)-/      -/
    def right_canonize_upto(self, interval=0, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        assert 0<=interval<=len(self)
        for site in range(len(self)-1, interval-1, -1):
            self.right_canonize_site(site, chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)

    def both_canonize(self, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        self.left_canonize_upto(chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)
        self.right_canonize_upto(chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)


    def is_left_canonical_site(self, site):
        S = self.bdts[site]
        V = self.tensors[site]
        SV = S*V
        return SV.is_left_unitary(self.get_right_labels_site(site-1)+self.get_phys_labels_site(site))

    def is_right_canonical_site(self, site):
        U = self.tensors[site]
        S = self.bdts[site+1]
        US = U*S
        return US.is_right_unitary(self.get_phys_labels_site(site)+self.get_left_labels_site(site+1))

    def is_left_canonical_upto(self, interval=None):
        if interval is None: interval = len(self)
        assert 0<=interval<=len(self)
        for site in range(interval):
            if not self.is_left_canonical_site(site):
                return False
        return True

    def is_right_canonical_upto(self, interval=0):
        assert 0<=interval<=len(self)
        for site in range(len(self)-1, interval-1, -1):
            if not self.is_right_canonical_site(site):
                return False
        return True

    def is_both_canonical(self, end_dealing="no"):
        if end_dealing=="no":
            return self.is_left_canonical_upto(len(self)-1) and self.is_right_canonical_upto(1)
        return self.is_left_canonical_upto() and self.is_right_canonical_upto()


    def to_tensor(self):
        re = copyModule.deepcopy(self.bdts[0])
        for i in range(len(self)):
            x = self.tensors[i]
            re *= x
            x = self.bdts[i+1]
            re *= x
        re.remove_all_dummy_indices()
        return re





# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DSimBTPS(Fin1DSimBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        Fin1DSimBTPS.__init__(self, tensors, bdts, phys_labelss=phys_labelss)

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


    def get_left_labels_site(self, site):
        site = site % len(self)
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)

    def get_right_labels_site(self, site):
        site = site % len(self)
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[(site+1)%len(self)].labels)

    # get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    def get_left_eigenvector(self): #TOCHUU
        temp = Tensor(1)
        in_bra_labels = self.get_right_labels_site(i)
        in_ket_labels = aster_labels(in_bra_labels)
        out_bra_labels = prime_labels(in_bra_labels)
        out_ket_labels = aster_labels(out_bra_labels)
        for i in range(len(self)):
            temp *= self.bdts[i]
            temp *= self.bdts[i].adjoint(self.get_left_labels_bond(i),self.get_right_labels_bond(i), style="aster")
            if i==len(self)-1:
                migiue = self.tensors[i].replace_labels(in_bra_labels, out_bra_labels, inplace=False)
                migisita = migiue.adjoint(self.get_left_labels_site(i), out_bra_labels, style="aster")
                temp *= migiue
                temp *= migisita
            else:
                temp *= self.tensors[i]
                temp *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")

