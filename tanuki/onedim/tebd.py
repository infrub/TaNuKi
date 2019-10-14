from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



class Obc1DTEBD:
    def __init__(self, psi, gates, chi=None, refresh_period=None, keep_universal_canonicality=True, gating_order="grissand"):
        self.psi = psi
        self.gates = gates
        self.chi = chi
        self.refresh_period = refresh_period
        self.keep_universal_canonicality = keep_universal_canonicality
        self.gating_order = gating_order

    def onesweep(self):
        apply_everyplace_fin1DSimBTPS_fin1DSimTPOs(self.psi, self.gates, chi=self.chi, keep_universal_canonicality=self.keep_universal_canonicality, gating_order=self.gating_order):