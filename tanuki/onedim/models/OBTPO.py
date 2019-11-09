from tanuki.onedim.models._mixins import *

#  ██████  ██████  ████████ ██████   ██████  
# ██    ██ ██   ██    ██    ██   ██ ██    ██ 
# ██    ██ ██████     ██    ██████  ██    ██ 
# ██    ██ ██   ██    ██    ██      ██    ██ 
#  ██████  ██████     ██    ██       ██████  
class Opn1DBTPO(MixinOpn1DBTP_):
    def __init__(self, tensors, bdts, physout_labelss, physin_labelss, is_unitary=False, is_hermite=False):
        self.tensors = list(tensors)
        self.bdts = list(bdts)
        if len(self.bdts)+1==len(self.tensors):
            self.bdts = [tni.dummy_diagonalTensor()] + self.bdts + [tni.dummy_diagonalTensor()]
        self.physout_labelss = list(physout_labelss)
        self.physin_labelss = list(physin_labelss)
        self.is_unitary = is_unitary
        self.is_hermite = is_hermite

    def __repr__(self):
        return f"Opn1DBTPO(tensors={self.tensors}, bdts={self.bdts}, physout_labelss={self.physout_labelss}, physin_labelss={self.physin_labelss}, is_unitary={self.is_unitary}, is_hermite={self.is_hermite})"

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
            bdt = self.bdts[len(self)]
            dataStr += str(bdt)
            dataStr += "\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = "[\n" + dataStr + "],\n"
        dataStr += f"physout_labelss={self.physout_labelss},\n"
        dataStr += f"physin_labelss={self.physin_labelss},\n"
        dataStr += f"is_unitary={self.is_unitary},\n"
        dataStr += f"is_hermite={self.is_hermite},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Opn1DBTPO(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return self.tensors.__len__()

    def to_tensor(self):
        return self.to_TPO().to_TMO().to_tensor()

    def to_TMO(self):
        return self.to_TPO().to_TMO()

    def to_TPO(self):
        tensors = []
        for i in range(len(self)):
            tensors.append( self.bdts[i][self.get_right_labels_bond(i)] * self.tensors[i][self.get_left_labels_site(i)] )
        tensors[-1] = tensors[-1][self.get_right_labels_site(len(self)-1)] * self.bdts[len(self)][self.get_left_labels_bond(len(self))]
        from tanuki.onedim.models.OTPO import Opn1DTPO
        return Opn1DTPO(tensors, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)

    def to_BTPO(self):
        if self.bdts[0].is_dummy() and self.bdts[-1].is_dummy():
            return self
        tensors = list(self.tensors)
        tensors[0] *= self.bdts[0]
        tensors[-1] *= self.bdts[-1]
        bdts = [tni.dummy_diagonalTensor()] + self.bdts[1:-1] + [tni.dummy_diagonalTensor()]
        return Opn1DBTPO(self.tensors, self.bdts, self.physout_labelss, self.physin_labelss)
