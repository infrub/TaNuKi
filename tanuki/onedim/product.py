from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



def apply_fin1DSimBTPS_fin1DSimTPO(mps, mpo, offset=0, chi=None, keep_universal_canonicality=True, keep_phys_labels=False):
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