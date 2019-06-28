import copy as copyModule
from tanuki import tensor_core as tnc
from tanuki.utils import *
from tanuki.errors import *
import textwrap



class TensorFrame(tnc.TensorLabelingMixin):
    def __init__(self, shape, labels):
        self.shape = shape
        self.ndim = len(shape)
        self.labels = labels

    def __repr__(self):
        return f"TensorFrame(shape={self.shape}, labels={self.labels})"

    def __str__(self):
        re = \
        f"TensorFrame(\n" + \
        f"    shape={self.shape},\n" + \
        f"    labels={self.labels},\n" + \
        f")"

        return re
    def __mul__(self, other):
        return tensorFrame_contract_common(self, other)[0]






def tensorFrame_contract_common_and_cost(A, B):
    commonLabels = intersection_list(A.labels, B.labels)
    aIndicesContract, aIndicesNotContract = A.normarg_complement_indices_back(commonLabels)
    bIndicesContract, bIndicesNotContract = B.normarg_complement_indices_front(commonLabels)
    aLabelsContract, aLabelsNotContract = A.labels_of_indices(aIndicesContract), A.labels_of_indices(aIndicesNotContract)
    bLabelsContract, bLabelsNotContract = B.labels_of_indices(bIndicesContract), B.labels_of_indices(bIndicesNotContract)
    aDimsContract, aDimsNotContract = A.dims(aIndicesContract), A.dims(aIndicesNotContract)
    bDimsContract, bDimsNotContract = B.dims(bIndicesContract), B.dims(bIndicesNotContract)
    assert aDimsContract == bDimsContract
    cLabels = aLabelsNotContract + bLabelsNotContract
    cDims = aDimsNotContract + bDimsNotContract
    elim = soujou(aDimsContract)
    cost = soujou(cDims)*elim
    return TensorFrame(cDims, cLabels), cost, elim


def tensor_to_tensorFrame(T):
    return TensorFrame(T.shape, T.labels)




def netcon(Ts, algorithm="brute"):
    if algorithm=="brute":
        from tanuki.netcon.netcon_brute import NetconBrute as NetconClass
    else:
        raise UndecidedError

    TFs = list(map(tensor_to_tensorFrame, Ts))
    c = NetconClass(TFs)
    return c.generate()

def contract_all_common(Ts, algorithm="brute"):
    return netcon(Ts, algorithm=algorithm)(*Ts)
