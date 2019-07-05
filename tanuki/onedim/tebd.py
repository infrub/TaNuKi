from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



class Fin1DSimTEBD:
    def __init__(self, psi, gates, chi=None, refresh_period=None, gating_order="grissand"): #="trill"
        self.psi = psi
        self.gates = gates
        self.chi = chi
        self.refresh_period = refresh_period
        self.gating_order = gating_order

    def onesweep(self):
