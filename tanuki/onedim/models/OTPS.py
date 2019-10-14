from tanuki.onedim.models._mixins import *

#  ██████  ████████ ██████  ███████ 
# ██    ██    ██    ██   ██ ██      
# ██    ██    ██    ██████  ███████ 
# ██    ██    ██    ██           ██ 
#  ██████     ██    ██      ███████ 
# tensors[0] -- tensors[1] -- ... -- tensors[len-1]
class Obc1DTPS(Mixin_1DSim_PS, MixinObc1DTP_):
    def __init__(self, tensors, phys_labelss=None):
        self.tensors = list(tensors)
        if phys_labelss is None:
            self.phys_labelss = [self.get_guessed_phys_labels_site(site) for site in range(len(self))]
        else:
            self.phys_labelss = list(phys_labelss)

    def __repr__(self):
        return f"Obc1DTPS(tensors={self.tensors}, phys_labelss={self.phys_labelss})"

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

        dataStr = f"Obc1DTPS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def adjoint(self):
        tensors = [self.get_bra_site(site) for site in range(len(self))]
        from tanuki.onedim.models.OTPS import Obc1DTPS
        return Obc1DTPS(tensors, self.phys_labelss)

    def to_tensor(self):
        return self.to_TMS().to_tensor()

    def to_TMS(self):
        t = 1
        for i in range(len(self)):
            t *= self.tensors[i]
        from tanuki.onedim.models.OTMS import Obc1DTMS
        return Obc1DTMS(t, self.phys_labelss)

    def to_TPS(self):
        return self

    def to_BTPS(self):
        bdts = [tni.dummy_diagonalTensor()]
        for i in range(1,len(self)):
            labels = self.get_left_labels_site(i)
            shape = self.get_left_shape_site(i)
            bdt = tni.identity_diagonalTensor(shape, labels)
            bdts.append(bdt)
        bdts.append(tni.dummy_diagonalTensor())
        from tanuki.onedim.models.OBTPS import Obc1DBTPS
        return Obc1DBTPS(self.tensors, bdts, self.phys_labelss)

