from tanuki.onedim.models._mixins import *

# ██ ████████ ██████  ███████ 
# ██    ██    ██   ██ ██      
# ██    ██    ██████  ███████ 
# ██    ██    ██           ██ 
# ██    ██    ██      ███████ 
# )-- tensors[0] -- tensors[1] -- ... -- tensors[len-1] --(
class Inf1DTPS(Mixin_1DSim_PS, MixinInf1DTP_):
    def __init__(self, tensors, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Inf1DTPS(tensors={self.tensors}, phys_labelss={self.phys_labelss})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                tensor = self.tensors[i]
                dataStr += str(tensor)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Inf1DTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        return Inf1DTPS(tensors, self.phys_labelss)

    def to_tensor(self): # TODO losts information
        re = 1
        for i in range(len(self)):
            re *= self.bdts[i]
        return re

    def to_TPS(self):
        return self

    def to_BTPS(self):
        bdts = []
        for i in range(len(self)):
            labels = self.get_left_labels_site(i)
            shape = self.get_left_shape_site(i)
            bdt = tni.identity_diagonalTensor(shape, labels)
            bdts.append(bdt)
        from tanuki.onedim.models.IBTPS import Inf1DBTPS
        return Inf1DBTPS(self.tensors, bdts, self.phys_labelss)
