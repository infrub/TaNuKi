import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from numpy import prod as soujou
import textwrap

#Finite 1D Simple Tensor Product State
class Fin1DSimTPS:
    def __init__(self, tensors, phys_labelss=None, left_ext_virt_labels=None, right_ext_virt_labels=None):
        self.tensors = tensors
        if left_ext_virt_labels is None:
            self.left_ext_virt_labels = []
        else: 
            self.left_ext_virt_labels = left_ext_virt_labels
        if right_ext_virt_labels is None:
            self.right_ext_virt_labels = []
        else:
            self.right_ext_virt_labels = right_ext_virt_labels
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = phys_labelss

    def __copy__(self):
        return Fin1DSimTPS(copyModule.copy(self.tensors), phys_labelss=copyModule.copy(self.phys_labelss), left_ext_virt_labels=copyModule.copy(left_ext_virt_labels), right_ext_virt_labels=copyModule.copy(right_ext_virt_labels))

    def __deepcopy__(self, memo=None):
        if memo is None: memo = {}
        return Fin1DSimTPS(copyModule.deepcopy(self.tensors), phys_labelss=copyModule.deepcopy(self.phys_labelss), left_ext_virt_labels=copyModule.deepcopy(self.left_ext_virt_labels), right_ext_virt_labels=copyModule.deepcopy(self.right_ext_virt_labels))

    def copy(self, shallow=False):
        if shallow:
            return self.__copy__()
        else:
            return self.__deepcopy__()


    def __repr__(self):
        return f"Fin1DSimTPS(tensors={self.tensors}, phys_labelss={self.phys_labelss}, left_ext_virt_labels={self.left_ext_virt_labels}, right_ext_virt_labels={self.right_ext_virt_labels})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for tensor in self.tensors:
                dataStr += str(tensor)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")
        dataStr = "[\n" + dataStr + "\n]"
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"Fin1DSimTPS(\n" + \
        dataStr + "\n" + \
        f"    phys_labelss={self.phys_labelss},\n" + \
        f"    left_ext_virt_labels={self.left_ext_virt_labels},\n" + \
        f"    right_ext_virt_labels={self.right_ext_virt_labels},\n" + \
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


    def get_left_labels_site(self, site):
        if site==0:
            return self.left_ext_virt_labels
        return tnc.intersection_list(self.tensors[site-1].labels, self.tensors[site].labels)

    def get_right_labels_site(self, site):
        if site==len(self)-1:
            return self.right_ext_virt_labels
        return tnc.intersection_list(self.tensors[site].labels, self.tensors[site+1].labels)

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
        for site in range(len(self)-1):
            dim = self.get_right_dim_site(site)
            label = self.get_right_labels_site(site)[0]
            bdt = tni.identity_tensor(dim, label)
            bdts.append(bdt)
        return Fin1DSimBTPS(fused_self.tensors, bdts, phys_labelss=fused_self.phys_labelss, left_ext_virt_labels=fused_self.left_ext_virt_labels, right_ext_virt_labels=fused_self.right_ext_virt_labels)

    def to_tensor(self):
        re = tnc.Tensor(1)
        for x in self.tensors:
            re *= x
        return re



# tensors[0] -- bdts[0] -- tensors[1] -- ... -- tensors[-1]
class Fin1DSimBTPS:
    def __init__(self, tensors, bdts, phys_labelss=None, left_ext_virt_labels=None, right_ext_virt_labels=None):
        self.tensors = tensors
        self.bdts = bdts
        if left_ext_virt_labels is None:
            self.left_ext_virt_labels = []
        else: 
            self.left_ext_virt_labels = left_ext_virt_labels
        if right_ext_virt_labels is None:
            self.right_ext_virt_labels = []
        else:
            self.right_ext_virt_labels = right_ext_virt_labels
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = phys_labelss



    def __repr__(self):
        return f"Fin1DSimTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss}, left_ext_virt_labels={self.left_ext_virt_labels}, right_ext_virt_labels={self.right_ext_virt_labels})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self.tensors)):
                tensor = self.tensors[i]
                dataStr += str(tensor)
                dataStr += ",\n"
                if i==len(self.tensors)-1:
                    break
                bdt = self.bdts[i]
                dataStr += str(bdt)
                dataStr += "\n"
        dataStr = textwrap.indent(dataStr, "    ")
        dataStr = "[\n" + dataStr + "\n]"
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"Fin1DSimBTPS(\n" + \
        dataStr + "\n" + \
        f"    phys_labelss={self.phys_labelss},\n" + \
        f"    left_ext_virt_labels={self.left_ext_virt_labels},\n" + \
        f"    right_ext_virt_labels={self.right_ext_virt_labels},\n" + \
        f")"

        return re


    def __len__(self):
        return self.tensors.__len__()


    def get_left_labels_site(self, site):
        if site==0:
            return self.left_ext_virt_labels
        return tnc.intersection_list(self.bdts[site-1].labels, self.tensors[site].labels)

    def get_right_labels_site(self, site):
        if site==len(self)-1:
            return self.right_ext_virt_labels
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site].labels)

    def get_phys_labels_site(self, site):
        return self.phys_labelss[site]

    def get_guessed_phys_labels_site(self, site):
        return diff_list(self.tensors[site].labels, self.get_left_labels_site(site)+self.get_right_labels_site(site))


    def get_left_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_left_labels_site(site)))

    def get_right_dim_site(self, site):
        return soujou(self.tensors[site].dims_of_labels(self.get_right_labels_site(site)))


    # (site=1):
    # /-(0)-[1]-      /-
    # |      |    ==  |
    # \-(0)-[1]-      \-
    def left_canonize_site(self, site, chi=None, relative_threshold=1e-14):
        if site==len(self)-1:
            return
        U, S, V = tnd.truncated_svd(self.tensors[site]*self.bdts[site], self.get_left_labels_site(site)+self.get_phys_labels_site(site), chi=chi, relative_threshold=relative_threshold)
        self.tensors[site] = U
        self.bdts[site] = S
        self.tensors[site+1] = V*self.tensors[site+1]

    # (site=4):
    # -[4]-(4)-\      -\
    #   |      |  ==   |
    # -[4]-(4)-/      -/
    def right_canonize_site(self, site, chi=None, relative_threshold=1e-14):
        if site==0:
            return
        U, S, V = tnd.truncated_svd(self.bdts[site-1]*self.tensors[site], self.get_right_labels_site(site-1), chi=chi, relative_threshold=relative_threshold)
        self.tensors[site-1] = self.tensors[site-1]*U
        self.bdts[site-1] = S
        self.tensors[site] = V

    # (interval=2):
    # /-[0]-      /-(0)-[1]-      /-
    # |  |    ==  |      |    ==  |
    # \-[0]-      \-(0)-[1]-      \-
    def left_canonize_upto(self, interval=None, chi=None, relative_threshold=1e-14):
        if interval is None: interval = len(self)
        assert 0<=interval<=len(self)
        for site in range(interval):
            self.left_canonize_site(site, chi=chi, relative_threshold=relative_threshold)

    # (interval=4):
    # -[4]-(4)-\      -[5]-\      -\
    #   |      |  ==    |  |  ==   |
    # -[4]-(4)-/      -[5]-/      -/
    def right_canonize_upto(self, interval=0, chi=None, relative_threshold=1e-14):
        assert 0<=interval<=len(self)
        for site in range(len(self)-1, interval-1, -1):
            self.right_canonize_site(site, chi=chi, relative_threshold=relative_threshold)

    def both_canonize(self, chi=None, relative_threshold=1e-14):
        self.left_canonize_upto(chi=chi, relative_threshold=relative_threshold)
        self.right_canonize_upto(chi=chi, relative_threshold=relative_threshold)


    def to_tensor(self):
        re = tnc.Tensor(1)
        for i in range(len(self)):
            x = self.tensors[i]
            re *= x
            if i==len(self)-1:
                break
            x = self.bdts[i]
            re *= x
        return re




# )-- tensors[0] -- bdts[0] -- tensors[1] -- ... -- tensors[-1] -- bdts[-1] --(
#class Inf1DSimBTPS:

