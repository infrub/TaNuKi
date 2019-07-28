from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



class Fin1DSimTEBD:
    def __init__(self, psi, gates, chi=None, refresh_period=None, gating_order="grissand", keep_universal_canonicality=True): #="trill"
        self.psi = psi
        self.gates = gates
        self.chi = chi
        self.refresh_period = refresh_period
        self.gating_order = gating_order
        self.keep_universal_canonicality = keep_universal_canonicality

    def onesweep(self):
        if gating_order == "grissand":
            for gate in self.gates:
                for k in range(len(self.psi)-len(gate)+1):
                    apply_fin1DSimBTPS_fin1DSimTPO(self.psi, gate, offset=k, chi=self.chi, keep_universal_canonicality=self.keep_universal_canonicality, keep_phys_labels=True)
        elif gating_order == "trill":
            for gate in self.gates:
                for i in range(len(gate)):
                    for j in range((len(self.psi) - i) // len(gate)):
                        k = j * len(gate) + i
                        apply_fin1DSimBTPS_fin1DSimTPO(self.psi, gate, offset=k, chi=self.chi, keep_universal_canonicality=self.keep_universal_canonicality, keep_phys_labels=True)


