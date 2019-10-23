from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.IBTPS import Inf1DBTPS

#  ██████ ██████  ████████ ██████  ███████ 
# ██      ██   ██    ██    ██   ██ ██      
# ██      ██████     ██    ██████  ███████ 
# ██      ██   ██    ██    ██           ██ 
#  ██████ ██████     ██    ██      ███████ 
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Cyc1DBTPS(Inf1DBTPS):
    def __init__(self, tensors, bdts, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        self.bdts = CyclicList(bdts)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Cyc1DBTPS(tensors={self.tensors}, bdts={self.bdts}, phys_labelss={self.phys_labelss})"

    def __str__(self, nodata=False):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += bdt.__str__(nodata=nodata)
                dataStr += "\n"
                tensor = self.tensors[i]
                dataStr += tensor.__str__(nodata=nodata)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Cyc1DBTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        bdts = [self.get_bra_bond(bondsite) for bondsite in range(len(self))]
        return Cyc1DBTPS(tensors, bdts, self.phys_labelss)

    def truncate(self, chi, normalize=True, algname="canonize"):
        if algname == "naive":
            if normalize:
                weight = 1.0
                for bde in range(len(self)):
                    U,S,V = self.bdts[bde].svd(self.get_left_labels_bond(bde), chi=chi, svd_labels=2)
                    weight *= S.normalize(inplace=True)
                    self.tensors[bde-1] = self.tensors[bde-1] * U
                    self.bdts[bde] = S
                    self.tensors[bde] = V * self.tensors[bde]
                return weight
            else:
                for bde in range(len(self)):
                    U,S,V = self.bdts[bde].svd(self.get_left_labels_bond(bde), chi=chi, svd_labels=2)
                    self.tensors[bde-1] = self.tensors[bde-1] * U
                    self.bdts[bde] = S
                    self.tensors[bde] = V * self.tensors[bde]
                return

        elif algname == "canonize":
            if normalize:
                memo = {}
                self.universally_canonize(chi=chi, transfer_normalize=True, memo=memo)
                weight = sqrt(memo["w"])
                return weight
            else:
                self.universally_canonize(chi=chi, transfer_normalize=False)
                return

        elif algname in ["iterative"]:
            pass
