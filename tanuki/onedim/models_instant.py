from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



def random_fin1DSimTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is not None and len(virt_labelss) == length-1:
        virt_labelss = [[]] + virt_labelss + [[]]
    if virt_dimss is not None and len(virt_dimss) == length-1:
        virt_dimss = [()] + virt_dimss + [()]

    if virt_labelss is None and virt_dimss is None:
        virt_labelss = [[]] + [[unique_label()] for _ in range(length-1)] + [[]]
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_dimss is None:
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_labelss is None:
        virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[site+1], virt_labelss[site]+phys_labelss[site]+virt_labelss[site+1] , dtype=dtype) )

    return Obc1DTPS(tensors, phys_labelss)


def random_fin1DSimBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is not None and len(virt_labelss) == length-1:
        virt_labelss = [[]] + virt_labelss + [[]]
    if virt_dimss is not None and len(virt_dimss) == length-1:
        virt_dimss = [()] + virt_dimss + [()]

    if virt_labelss is None and virt_dimss is None:
        virt_labelss = [[]] + [[unique_label()] for _ in range(length-1)] + [[]]
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_dimss is None:
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_labelss is None:
        virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]

    bdts = []
    for bondsite in range(length+1):
        if len(virt_dimss[bondsite])==0:
            bdts.append( tni.dummy_diagonalTensor() )
        else:
            bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[site+1], virt_labelss[site]+phys_labelss[site]+virt_labelss[site+1] , dtype=dtype) )

    return Obc1DBTPS(tensors, bdts, phys_labelss)



def random_inf1DSimBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None and virt_dimss is None:
        virt_labelss = [[unique_label()] for _ in range(length)]
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_dimss is None:
        virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
    elif virt_labelss is None:
        virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]

    bdts = []
    for bondsite in range(length):
        if len(virt_dimss[bondsite])==0:
            bdts.append( tni.dummy_diagonalTensor() )
        else:
            bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Inf1DBTPS(tensors, bdts, phys_labelss)

