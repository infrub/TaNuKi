from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



class NonBridgeBondEnv:
    def __init__(self,tensor,ket_left_labels,ket_right_labels,bra_left_labels,bra_right_labels):
        self.tensor = tensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = bra_left_labels
        self.bra_right_labels = bra_right_labels


# If a bond in TN is a bridge, when remove the bond, the TN is disconnected into left and right
class BridgeBondEnv:
    def __init__(self,leftTensor,rightTensor,ket_left_labels,ket_right_labels,bra_left_labels,bra_right_labels):
        self.leftTensor = leftTensor
        self.rightTensor = rightTensor
        self.ket_left_labels = ket_left_labels
        self.ket_right_labels = ket_right_labels
        self.bra_left_labels = bra_left_labels
        self.bra_right_labels = bra_right_labels

