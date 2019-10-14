from tanuki.onedim.models._mixins import *

# ██ ████████ ██████   ██████  
# ██    ██    ██   ██ ██    ██ 
# ██    ██    ██████  ██    ██ 
# ██    ██    ██      ██    ██ 
# ██    ██    ██       ██████  
# )-- tensors[0] -- tensors[1] -- ... -- tensors[len-1] --(
class Inf1DTPO(Mixin_1DSim_PO, MixinInf1DTP_):
    def __init__(self, tensors, physout_labelss, physin_labelss, is_unitary=False, is_hermite=False):
        self.tensors = CyclicList(tensors)
        self.physout_labelss = CyclicList(physout_labelss)
        self.physin_labelss = CyclicList(physin_labelss)
        self.is_unitary = is_unitary
        self.is_hermite = is_hermite

    def __repr__(self):
        return f"Inf1DTPO(tensors={self.tensors}, physout_labelss={self.physout_labelss}, physin_labelss={self.physin_labelss}, is_unitary={self.is_unitary}, is_hermite={self.is_hermite})"

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

        dataStr = f"Inf1DTPO(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def to_TPO(self):
        return self

    def to_BTPO(self):
        bdts = []
        for i in range(len(self)):
            labels = self.get_left_labels_site(i)
            shape = self.get_left_shape_site(i)
            bdt = tni.identity_diagonalTensor(shape, labels)
            bdts.append(bdt)
        from tanuki.onedim.models.IBTPO import Inf1DBTPO
        return Inf1DBTPO(self.tensors, bdts, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)

