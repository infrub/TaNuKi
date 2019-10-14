from tanuki.onedim.models._mixins import *

#  ██████  ████████ ███    ███  ██████  
# ██    ██    ██    ████  ████ ██    ██ 
# ██    ██    ██    ██ ████ ██ ██    ██ 
# ██    ██    ██    ██  ██  ██ ██    ██ 
#  ██████     ██    ██      ██  ██████  
class Obc1DTMO: #Tensor Mass Operator
    def __init__(self, tensor, physout_labelss, physin_labelss, is_unitary=False, is_hermite=False):
        self.tensor = tensor
        self.physout_labelss = list(physout_labelss)
        self.physin_labelss = list(physin_labelss)
        self.is_unitary = is_unitary
        self.is_hermite = is_hermite

    def __repr__(self):
        return f"Obc1DTMO(tensor={self.tensor}, physout_labelss={self.physout_labelss}, physin_labelss={self.physin_labelss}, is_unitary={self.is_unitary}, is_hermite={self.is_hermite})"

    def __str__(self):
        dataStr = f"{self.tensor},\n"
        dataStr += f"physout_labelss={self.physout_labelss},\n"
        dataStr += f"physin_labelss={self.physin_labelss},\n"
        dataStr += f"is_unitary={self.is_unitary},\n"
        dataStr += f"is_hermite={self.is_hermite},\n"
        dataStr = textwrap.indent(dataStr, "    ")

        dataStr = f"Obc1DTMO(\n" + dataStr + f")"

        return dataStr

    def __len__(self):
        return len(self.physout_labelss)

    def to_tensor(self):
        return self.tensor

    def to_TMO(self):
        return self

    def to_TPO(self):
        return self.to_BTPO().to_TPO()

    def to_BTPO(self, chi=None, rtol=None, atol=None):
        G = self.tensor.copy()
        rev_tensors = []
        rev_bdts = []
        for i in range(len(self)-1,0,-1):
            G, b, a = tnd.truncated_svd(G, sum(self.physout_labelss[:i]+self.physin_labelss[:i],[]), chi=chi, rtol=rtol, atol=atol)
            rev_tensors.append(a)
            rev_bdts.append(b)
        rev_tensors.append(G)
        rev_tensors.reverse()
        rev_bdts.reverse()
        from tanuki.onedim.models.OBTPO import Obc1DBTPO
        return Obc1DBTPO(rev_tensors, rev_bdts, self.physout_labelss, self.physin_labelss, is_unitary=self.is_unitary, is_hermite=self.is_hermite)

    def exp(self, coeff=1):
        G = self.tensor
        V,W,Vh = tnd.tensor_eigh(G, sum(self.physout_labelss, []), sum(self.physin_labelss, []))
        W = W.exp(coeff)
        G = V*W*Vh
        is_hermite = False
        is_unitary = False
        if self.is_hermite and np.real(coeff)==0:
            is_unitary = True
        if self.is_hermite and np.imag(coeff)==0:
            is_hermite = True
        return Obc1DTMO(G, self.physout_labelss, self.physin_labelss, is_unitary=is_unitary, is_hermite=is_hermite)


