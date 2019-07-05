from tanuki import *
from tanuki.onedim import *

class OneDimGate:
    def __init__(self, tensor, physout_labelss, physin_labelss):
        self.tensor = tensor
        self.physout_labelss = physout_labelss
        self.physin_labelss = physin_labelss

    def __repr__(self):
        return f"OneDimGate({self.tensor}, {self.physout_labelss}, {self.physin_labelss})"

    def __str__(self):
        dataStr = f"{self.tensor},\n"
        dataStr += f"physout_labelss={self.physout_labelss},\n"
        dataStr += f"physin_labelss={self.physin_labelss},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"OneDimGate(\n" + \
        dataStr + \
        f")"

        return re


    def to_fin1DSimBTPO(self):
        G = self.tensor.copy()
        tensors = []
        bdts = []
        for physout_labels, physin_labels in zip(self.physout_labelss, self.physin_labelss):
            a, b, G = tensor_svd(G, physout_labels+physin_labels)
            tensors.append(a)
            bdts.append(b)
        tensors.append(G)
        return Fin1DSimBTPO(tensors, bdts, self.physout_labelss, self.physin_labelss)

    def exp(self, coeff=1):
        G = self.tensor
        V,W,Vh = tensor_eigh(G, sum(self.physout_labelss, []), sum(self.physin_labelss, []))
        W = W.exp(coeff)
        G = V*W*Vh
        return OneDimGate(G, self.physout_labelss, self.physin_labelss)