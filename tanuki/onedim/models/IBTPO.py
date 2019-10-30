from tanuki.onedim.models._mixins import *
from tanuki.onedim.models.OBTPO import Opn1DBTPO

# ██ ██████  ████████ ██████   ██████  
# ██ ██   ██    ██    ██   ██ ██    ██ 
# ██ ██████     ██    ██████  ██    ██ 
# ██ ██   ██    ██    ██      ██    ██ 
# ██ ██████     ██    ██       ██████  
# )-- bdts[0] -- tensors[0] -- bdts[1] -- tensors[1] -- ... -- bdts[-1] -- tensors[-1] --(
class Inf1DBTPO(MixinInf1DBTP_, Opn1DBTPO):
    def __init__(self, tensors, bdts, physout_labelss, physin_labelss, is_unitary=False, is_hermite=False):
        self.tensors = CyclicList(tensors)
        self.bdts = CyclicList(bdts)
        self.physout_labelss = CyclicList(physout_labelss)
        self.physin_labelss = CyclicList(physin_labelss)
        self.is_unitary = is_unitary
        self.is_hermite = is_hermite

    def __repr__(self):
        return f"Inf1DBTPO(tensors={self.tensors}, bdts={self.bdts}, physout_labelss={self.physout_labelss}, physin_labelss={self.physin_labelss}, is_unitary={self.is_unitary}, is_hermite={self.is_hermite})"

    def __str__(self):
        if len(self) > 20:
            dataStr = " ... "
        else:
            dataStr = ""
            for i in range(len(self)):
                bdt = self.bdts[i]
                dataStr += str(bdt)
                dataStr += "\n"
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

        dataStr = f"Inf1DBTPO(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def to_TPO(self):
        tensors = []
        for i in range(len(self)):
            tensors.append( self.bdts[i][self.get_right_labels_bond(i)] * self.tensors[i][self.get_left_labels_site(i)] )
        from tanuki.onedim.models.OTPO import Opn1DTPO
        return Inf1DTPO(tensors, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)

    def to_BTPO(self):
        return self