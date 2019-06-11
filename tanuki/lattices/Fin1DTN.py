import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
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
            bdt = tni.identity_diagonalTensor(dim, label)
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


    def get_left_shape_site(self, site):
        return self.tensors[site].dims_of_labels(self.get_left_labels_site(site))

    def get_right_shape_site(self, site):
        return self.tensors[site].dims_of_labels(self.get_right_labels_site(site))

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
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
            self.tensors[site] = U/self.bdts[site]
            self.bdts[site+1] = S
            return V

        else:
            return

    # (site=1):
    # /-(1)-[1]-      /-
    # |      |    ==  |
    # \-(1)-[1]-      \-
    def left_canonize_not_end_site(self, site, chi=None, relative_threshold=1e-14):
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
        self.tensors[site] = U/self.bdts[site]
        self.bdts[site+1] = S
        self.tensors[site+1] = V*self.tensors[site+1]

    def left_canonize_site(self, site, chi=None, relative_threshold=1e-14, end_dealing="normalize"):
        if site==len(self)-1:
            self.left_canonize_right_end(chi=chi, relative_threshold=relative_threshold, end_dealing=end_dealing)
        else:
            self.left_canonize_not_end_site(site, chi=chi, relative_threshold=relative_threshold)


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
            U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site), chi=chi, relative_threshold=relative_threshold)
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
            return
        U, S, V = tnd.truncated_svd(self.bdts[site]*self.tensors[site]*self.bdts[site+1], self.get_left_labels_bond(site), chi=chi, relative_threshold=relative_threshold)
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
        re = SV.is_left_unitary(self.get_right_labels_site(site-1)+self.get_phys_labels_site(site))
        print(site,re)
        return re

    def is_right_canonical_site(self, site):
        U = self.tensors[site]
        S = self.bdts[site+1]
        US = U*S
        re = US.is_right_unitary(self.get_phys_labels_site(site)+self.get_left_labels_site(site+1))
        print(site,re)
        return re

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
        if type(tensors) != CyclicList:
            tensors = CyclicList(tensors)
        if type(bdts) != CyclicList:
            bdts = CyclicList(bdts)
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
            x = self.bdts[i]
            re *= x
            x = self.tensors[i]
            re *= x
        re.remove_all_dummy_indices()
        return re


    # get L s.t.
    # /-(0)-[0]-...-(len-1)-[len-1]-          /-
    # L      |                 |      ==  c * L
    # \-(0)-[0]-...-(len-1)-[len-1]-          \-
    def get_left_eigenvector(self): #TOCHUU
        label_base = "TFL" #unique_label()
        inbra = label_base + "_inbra"
        inket = label_base + "_inket"
        outbra = label_base + "_outbra"
        outket = label_base + "_outket"
        dim = self.get_right_dim_site(len(self)-1)
        shape = self.get_right_shape_site(len(self)-1)
        rawl = self.get_right_labels_site(len(self)-1)

        TF_L = tni.identity_tensor(dim, shape, labels=[inket]+rawl)
        TF_L *= tni.identity_tensor(dim, shape, labels=[inbra]+aster_labels(rawl))
        for i in range(len(self)):
            TF_L *= self.bdts[i]
            TF_L *= self.bdts[i].adjoint(self.get_left_labels_bond(i),self.get_right_labels_bond(i), style="aster")
            TF_L *= self.tensors[i]
            TF_L *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")
        TF_L *= tni.identity_tensor(dim, shape, labels=[outket]+rawl)
        TF_L *= tni.identity_tensor(dim, shape, labels=[outbra]+aster_labels(rawl))
        #print("TF_L: ", TF_L.trace(inbra, inket))

        w_L, V_L = tnd.tensor_eigsh(TF_L, [outket,outbra], [inket,inbra])
        V_L.hermite(inket, inbra, assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_L.split_index(inket, shape, rawl)
        V_L.split_index(inbra, shape, aster_labels(rawl))

        return V_L

    # get R s.t.
    # -[0]-...-(len-1)-[len-1]-(0)-\          -\
    #   |                 |        R  ==  c *  R
    # -[0]-...-(len-1)-[len-1]-(0)-/          -/
    def get_right_eigenvector(self): #TOCHUU
        label_base = "TFR" #unique_label()
        inbra = label_base + "_inbra"
        inket = label_base + "_inket"
        outbra = label_base + "_outbra"
        outket = label_base + "_outket"
        dim = self.get_left_dim_site(0)
        shape = self.get_left_shape_site(0)
        rawl = self.get_left_labels_site(0)

        TF_R = tni.identity_tensor(dim, shape, labels=[inket]+rawl)
        TF_R *= tni.identity_tensor(dim, shape, labels=[inbra]+aster_labels(rawl))
        for i in range(len(self)-1, -1, -1):
            TF_R *= self.bdts[i+1]
            TF_R *= self.bdts[i+1].adjoint(self.get_left_labels_bond(i+1),self.get_right_labels_bond(i+1), style="aster")
            TF_R *= self.tensors[i]
            TF_R *= self.tensors[i].adjoint(self.get_left_labels_site(i),self.get_right_labels_site(i), style="aster")
        TF_R *= tni.identity_tensor(dim, shape, labels=[outket]+rawl)
        TF_R *= tni.identity_tensor(dim, shape, labels=[outbra]+aster_labels(rawl))

        w_R, V_R = tnd.tensor_eigsh(TF_R, [outket,outbra], [inket,inbra])
        V_R.hermite(inbra, inket, assume_definite_and_if_negative_then_make_positive=True, inplace=True)
        V_R.split_index(inket, shape, rawl)
        V_R.split_index(inbra, shape, aster_labels(rawl))
        
        return V_R

    # ref: https://arxiv.org/abs/0711.3960
    def canonize_end(self, chi=None, relative_threshold=1e-14):
        dl_label = unique_label()
        dr_label = unique_label()
        V_L = self.get_left_eigenvector()
        V_R = self.get_right_eigenvector()
        #print("V_L:", V_L)
        #print("V_R:", V_R)
        Yh, d_L, Y = tnd.tensor_eigh(V_L, self.get_right_labels_site(len(self)-1), aster_labels(self.get_right_labels_site(len(self)-1)), eigh_labels=dl_label)
        #print("sahen", tensou*Yh*d_L*Y)
        #print("uhen", Yh*d_L*Y)
        Y.unaster_labels(aster_labels(self.get_right_labels_site(len(self)-1)))
        #print(Yh*Y)
        X, d_R, Xh = tnd.tensor_eigh(V_R, self.get_left_labels_site(0), aster_labels(self.get_left_labels_site(0)), eigh_labels=dr_label)
        Xh.unaster_labels(aster_labels(self.get_left_labels_site(0)))
        #print(Xh*X)
        l0 = self.bdts[0]
        G = d_L.sqrt() * Yh * l0 * X * d_R.sqrt()
        U, S, V = tnd.truncated_svd(G, dl_label, dr_label, chi=chi, relative_threshold=relative_threshold)
        M = Y * d_L.inv().sqrt() * U
        N = V * d_R.inv().sqrt() * Xh
        # l0 == M*S*N
        self.bdts[0] = S
        self.tensors[0] = N * self.tensors[0]
        self.tensors[len(self)-1] = self.tensors[len(self)-1] * M

    left_canonize_site = Fin1DSimBTPS.left_canonize_not_end_site

    def canonize(self, chi=None, relative_threshold=1e-14):
        self.canonize_end(chi=chi, relative_threshold=relative_threshold)
        self.left_canonize_site(0, chi=chi, relative_threshold=relative_threshold)
        self.left_canonize_site(1, chi=chi, relative_threshold=relative_threshold)
        self.left_canonize_site(2, chi=chi, relative_threshold=relative_threshold)
