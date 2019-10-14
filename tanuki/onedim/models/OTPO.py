from tanuki.onedim.models._mixins import *

#  ██████  ████████ ██████   ██████  
# ██    ██    ██    ██   ██ ██    ██ 
# ██    ██    ██    ██████  ██    ██ 
# ██    ██    ██    ██      ██    ██ 
#  ██████     ██    ██       ██████  
class Obc1DTPO(Mixin_1DSim_PO, MixinObc1DTP_):
    def __init__(self, tensors, physout_labelss, physin_labelss, is_unitary=False, is_hermite=False):
        self.tensors = list(tensors)
        self.physout_labelss = list(physout_labelss)
        self.physin_labelss = list(physin_labelss)
        self.is_unitary = is_unitary
        self.is_hermite = is_hermite

    def __repr__(self):
        return f"Obc1DTPO(tensors={self.tensors}, physout_labelss={self.physout_labelss}, physin_labelss={self.physin_labelss}, is_unitary={self.is_unitary}, is_hermite={self.is_hermite})"

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
        dataStr += f"physout_labelss={self.physout_labelss},\n"
        dataStr += f"physin_labelss={self.physin_labelss},\n"
        dataStr += f"is_unitary={self.is_unitary},\n"
        dataStr += f"is_hermite={self.is_hermite},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Obc1DTPO(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()



    def to_tensor(self):
        return self.to_TMO().to_tensor()

    def to_TMO(self):
        t = 1
        for i in range(len(self)):
            t *= self.tensors[i]
        from tanuki.onedim.models.OTMO import Obc1DTMO
        return Obc1DTMO(t, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)

    def to_TPO(self):
        return self

    def to_BTPO(self):
        bdts = [tni.dummy_diagonalTensor()]
        for i in range(1,len(self)):
            labels = self.get_left_labels_site(i)
            shape = self.get_left_shape_site(i)
            bdt = tni.identity_diagonalTensor(shape, labels)
            bdts.append(bdt)
        bdts.append(tni.dummy_diagonalTensor())
        from tanuki.onedim.models.OBTPO import Obc1DBTPO
        return Obc1DBTPO(self.tensors, bdts, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)
