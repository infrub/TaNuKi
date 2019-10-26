from tanuki.onedim.models._mixins import *

#  ██████ ████████ ███    ███ ███████ 
# ██         ██    ████  ████ ██      
# ██         ██    ██ ████ ██ ███████ 
# ██         ██    ██  ██  ██      ██ 
#  ██████    ██    ██      ██ ███████ 
class Cyc1DTMS: #Cyclic boundary Tensor Mass State
    def __init__(self, tensor, phys_labelss):
        self.tensor = tensor
        self.phys_labelss = CyclicList(phys_labelss)

    def __repr__(self):
        return f"Cyc1DTMS(tensor={self.tensor}, phys_labelss={self.physout_labelss})"

    def __str__(self):
        dataStr = f"{self.tensor},\n"
        dataStr += f"phys_labelss={self.phys_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Cyc1DTMS(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return len(self.phys_labelss)

    def __eq__(self, other):
        return self.tensor.move_all_indices(sum(self.phys_labelss,[])) == other.tensor.move_all_indices(sum(other.phys_labelss,[]))
