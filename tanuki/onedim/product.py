from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *
from tanuki.tnxp import xp



def inner_product_fin1DSimTPS_fin1DSimTPS(mps1, mps2):
    G = mps1.get_bra_site(0)[mps1.get_phys_labels_site(0)] * mps2.get_ket_site(0)[mps2.get_phys_labels_site(0)]
    for i in range(1,len(mps1)):
        G = G[mps1.get_bra_right_labels_site(i-1)] * mps1.get_bra_site(i)[mps1.get_bra_left_labels_site(i)]
        G = G[mps2.get_ket_right_labels_site(i-1)+mps1.get_phys_labels_site(i)] * mps2.get_ket_site(i)[mps2.get_ket_left_labels_site(i)+mps2.get_phys_labels_site(i)]
    return G.to_scalar()

def norm_fin1DSimTPS(mps):
    return xp.real(inner_product_fin1DSimTPS_fin1DSimTPS(mps,mps))

def abs_sub_fin1DSimTPS_fin1DSimTPS(mps1, mps2): # |mps1-mps2|
    return xp.sqrt( norm_fin1DSimTPS(mps1) + norm_fin1DSimTPS(mps2) - 2 * xp.real( inner_product_fin1DSimTPS_fin1DSimTPS(mps1,mps2) ) )




def apply_fin1DSimBTPS_fin1DSimTPO(mps, mpo, offset=0, chi=None, keep_universal_canonicality=True, keep_phys_labels=True):
    if keep_phys_labels:
        keeping_phys_labels = [mps.phys_labelss[offset+i] for i in range(len(mpo))]

    for i in range(len(mpo)):
        mps.tensors[offset+i] = mps.tensors[offset+i][mps.phys_labelss[offset+i]] * mpo.tensors[i][mpo.physin_labelss[i]]
        mps.phys_labelss[offset+i] = copyModule.copy(mpo.physout_labelss[i])
    for i in range(len(mpo)+1):
        mps.bdts[offset+i] = mps.bdts[offset+i] * mpo.bdts[i]

    if mpo.is_unitary or not keep_universal_canonicality:
        mps.universally_canonize(offset, offset+len(mpo))
    else:
        mps.universally_canonize()
    
    if chi is not None:
        if not mpo.is_unitary and not keep_universal_canonicality:
            logging.warn("apply_fin1DSimBTPS_fin1DSimTPO: Execute not optimal truncation.")

        for i in range(offset+1,offset+len(mpo)):
            labels = mps.get_left_labels_bond(i)
            assert len(labels) == 1
            label = labels[0]
            mps.tensors[i-1].truncate_index(label, chi, inplace=True)
            mps.bdts[i].truncate_index(label, chi, inplace=True)

            labels = mps.get_right_labels_bond(i)
            assert len(labels) == 1
            label = labels[0]
            mps.bdts[i].truncate_index(label, chi, inplace=True)
            mps.tensors[i].truncate_index(label, chi, inplace=True)

    if keep_phys_labels:
        for i in range(len(mpo)):
            mps.replace_phys_labels_site(offset+i, keeping_phys_labels[i])


def apply_everyplace_fin1DSimBTPS_fin1DSimTPOs(psi, gates, chi=None, keep_universal_canonicality=keep_universal_canonicality, gating_order="grissand"):
    if gating_order == "grissand":
        for gate in gates:
            for k in range(len(psi)-len(gate)+1):
                apply_fin1DSimBTPS_fin1DSimTPO(psi, gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)
    elif gating_order == "trill":
        for gate in self.gates:
            for i in range(len(gate)):
                for j in range((len(self.psi) - i) // len(gate)):
                    k = j * len(gate) + i
                    apply_fin1DSimBTPS_fin1DSimTPO(psi, gate, offset=k, chi=chi, keep_universal_canonicality=keep_universal_canonicality, keep_phys_labels=True)




