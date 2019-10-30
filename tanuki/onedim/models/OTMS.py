from tanuki.onedim.models._mixins import *

#  ██████  ████████ ███    ███ ███████ 
# ██    ██    ██    ████  ████ ██      
# ██    ██    ██    ██ ████ ██ ███████ 
# ██    ██    ██    ██  ██  ██      ██ 
#  ██████     ██    ██      ██ ███████ 
class Opn1DTMS: #Open boundary Tensor Mass State
    def __init__(self, tensor, phys_labelss):
        self.tensor = tensor
        self.phys_labelss = list(phys_labelss)

    def __repr__(self):
        return f"Opn1DTMS(tensor={self.tensor}, phys_labelss={self.physout_labelss})"

    def __str__(self):
        dataStr = f"{self.tensor},\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Opn1DTMS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return len(self.phys_labelss)

    def to_tensor(self):
        return self.tensor

    def to_TMS(self):
        return self

    def to_TPS(self):
        return self.to_BTPS().to_TPS()

    def to_BTPS(self):
        G = self.tensor.copy()
        rev_tensors = []
        rev_bdts = []
        for i in range(len(self)-1,0,-1):
            G, b, a = tnd.tensor_svd(G, sum(self.phys_labelss[:i],[]))
            rev_tensors.append(a)
            rev_bdts.append(b)
        rev_tensors.append(G)
        rev_tensors.reverse()
        rev_bdts.reverse()
        from tanuki.onedim.models.OBTPS import Opn1DBTPS
        return Opn1DBTPS(rev_tensors, rev_bdts, self.phys_labelss)

    def __eq__(self, other):
        return self.tensor.move_all_indices(sum(self.phys_labelss,[])) == other.tensor.move_all_indices(sum(other.phys_labelss,[]))
