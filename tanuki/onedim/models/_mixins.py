import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.tnxp import xp as xp
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim import product as tnop
import textwrap
from math import sqrt
import numpy as np
import warnings




class Mixin_1DSim_PS:
    def get_left_shape_site(self, site):
        return self.tensors[site].dims(self.get_left_labels_site(site))

    def get_right_shape_site(self, site):
        return self.tensors[site].dims(self.get_right_labels_site(site))

    def get_phys_labels_site(self, site):
        return self.phys_labelss[site]

    def get_guessed_phys_labels_site(self, site):
        return diff_list(self.tensors[site].labels, self.get_left_labels_site(site)+self.get_right_labels_site(site))

    def replace_phys_labels_site(self, site, labels):
        self.tensors[site].replace_labels(self.phys_labelss[site], labels)
        self.phys_labelss[site] = copyModule.copy(labels)

    def get_ket_site(self, site):
        return self.tensors[site].copy(shallow=True)

    def get_bra_site(self, site):
        return self.tensors[site].adjoint(self.get_left_labels_site(site), self.get_right_labels_site(site), style="aster")

    def get_ket_left_labels_site(self, site):
        return self.get_left_labels_site(site)

    def get_ket_right_labels_site(self, site):
        return self.get_right_labels_site(site)

    def get_bra_left_labels_site(self, site):
        return aster_labels(self.get_left_labels_site(site))

    def get_bra_right_labels_site(self, site):
        return aster_labels(self.get_right_labels_site(site))



class Mixin_1DSim_PO:
    def get_left_shape_site(self, site):
        return self.tensors[site].dims(self.get_left_labels_site(site))

    def get_right_shape_site(self, site):
        return self.tensors[site].dims(self.get_right_labels_site(site))

    def get_physout_labels_site(self, site):
        return self.physout_labelss[site]

    def get_physin_labels_site(self, site):
        return self.physin_labelss[site]

    def replace_physout_labels_site(self, site, labels):
        self.tensors[site].replace_labels(self.physout_labelss[site], labels)
        self.physout_labelss[site] = copyModule.copy(labels)

    def replace_physin_labels_site(self, site, labels):
        self.tensors[site].replace_labels(self.physin_labelss[site], labels)
        self.physin_labelss[site] = copyModule.copy(labels)





class Mixin_1DSimBTPS:
    def get_ket_bond(self, bondsite):
        return self.bdts[bondsite].copy(shallow=True)

    def get_bra_bond(self, bondsite):
        return self.bdts[bondsite].adjoint(self.get_left_labels_bond(bondsite), self.get_right_labels_bond(bondsite), style="aster")

    def get_ket_left_labels_bond(self, bondsite):
        return self.get_left_labels_bond(bondsite)

    def get_ket_right_labels_bond(self, bondsite):
        return self.get_right_labels_bond(bondsite)

    def get_bra_left_labels_bond(self, bondsite):
        return aster_labels(self.get_left_labels_bond(bondsite))

    def get_bra_right_labels_bond(self, bondsite):
        return aster_labels(self.get_right_labels_bond(bondsite))





class MixinObc1DTP_:
    def get_left_labels_site(self, site):
        if site==0:
            return []
        if site==len(self):
            return []
        return tnc.intersection_list(self.tensors[site-1].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        if site==-1:
            return []
        if site==len(self)-1:
            return []
        return tnc.intersection_list(self.tensors[site].labels, self.tensors[site+1].labels)



class MixinObc1DBTP_:
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



class MixinInf1DTP_:
    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site-1].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.tensors[site+1].labels)



class MixinInf1DBTP_:
    def get_left_labels_site(self, site):
        return tnc.intersection_list(self.bdts[site].labels, self.tensors[site].labels)
    def get_right_labels_site(self, site):
        return tnc.intersection_list(self.tensors[site].labels, self.bdts[site+1].labels)
    def get_left_labels_bond(self, bondsite):
        return self.get_right_labels_site(bondsite-1)
    def get_right_labels_bond(self, bondsite):
        return self.get_left_labels_site(bondsite)

