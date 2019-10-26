from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.ITPS import Inf1DTPS

#  ██████ ████████ ██████  ███████ 
# ██         ██    ██   ██ ██      
# ██         ██    ██████  ███████ 
# ██         ██    ██           ██ 
#  ██████    ██    ██      ███████ 
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Cyc1DTPS(Inf1DTPS):
    def __init__(self, tensors, phys_labelss=None):
        self.tensors = CyclicList(tensors)
        if phys_labelss is None:
            phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Cyc1DTPS(tensors={self.tensors}, phys_labelss={self.phys_labelss})"

    def __str__(self, nodata=False):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                tensor = self.tensors[i]
                dataStr += tensor.__str__(nodata=nodata)
                dataStr += ",\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Cyc1DTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        return Cyc1DTPS(tensors, self.phys_labelss)

    def to_TMS(self):
        result = self.tensors[0]
        for i in range(1,len(self)):
            result = result[self.get_right_labels_site(i-1)] * self.tensors[i][self.get_left_labels_site(i)]
        result.trace(self.get_left_labels_site(0),self.get_right_labels_site(len(self)-1))
        from tanuki.onedim.models.CTMS import Cyc1DTMS
        return Cyc1DTMS(result, self.phys_labelss)

    def to_TPS(self):
        return self

    def to_BTPS(self):
        bdts = []
        for i in range(len(self)):
            labels = self.get_left_labels_site(i)
            shape = self.get_left_shape_site(i)
            bdt = tni.identity_diagonalTensor(shape, labels)
            bdts.append(bdt)
        from tanuki.onedim.models.CBTPS import Cyc1DBTPS
        return Cyc1DBTPS(self.tensors, bdts, self.phys_labelss)






