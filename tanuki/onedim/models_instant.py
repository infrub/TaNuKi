from tanuki import tensor_core as tnc
from tanuki import tensor_instant as tni
from tanuki import decomp as tnd
from tanuki.utils import *
from tanuki.errors import *
from tanuki.onedim.models import *



def random_opn1DTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is not None and len(virt_labelss) == length-1:
        virt_labelss = [[]] + virt_labelss + [[]]
    if virt_dimss is not None and len(virt_dimss) == length-1:
        virt_dimss = [()] + virt_dimss + [()]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[]] + [[unique_label()] for _ in range(length-1)] + [[]]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[site+1], virt_labelss[site]+phys_labelss[site]+virt_labelss[site+1] , dtype=dtype) )

    return Opn1DTPS(tensors, phys_labelss)


def random_opn1DBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is not None and len(virt_labelss) == length-1:
        virt_labelss = [[]] + virt_labelss + [[]]
    if virt_dimss is not None and len(virt_dimss) == length-1:
        virt_dimss = [()] + virt_dimss + [()]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[unique_label()] for _ in range(length)]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]

    bdts = []
    for bondsite in range(length+1):
        bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[site+1], virt_labelss[site]+phys_labelss[site]+virt_labelss[site+1] , dtype=dtype) )

    return Opn1DBTPS(tensors, bdts, phys_labelss)



def random_inf1DTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[unique_label()] for _ in range(length)]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Inf1DTPS(tensors, phys_labelss)


def random_inf1DBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[unique_label()] for _ in range(length)]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]

    bdts = []
    for bondsite in range(length):
        bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Inf1DBTPS(tensors, bdts, phys_labelss)



def random_cyc1DTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[unique_label()] for _ in range(length)]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Cyc1DTPS(tensors, phys_labelss)


    
def random_cyc1DBTPS(phys_labelss, phys_dimss=None, virt_labelss=None, virt_dimss=None, phys_dim=2, chi=3, dtype=complex):
    phys_labelss = [p if type(p)==list else [p] for p in phys_labelss]
    length = len(phys_labelss)
    if phys_dimss is None:
        phys_dimss = [(phys_dim,)*len(phys_labels) for phys_labels in phys_labelss]

    if virt_labelss is None:
        if virt_dimss is None:
            virt_labelss = [[unique_label()] for _ in range(length)]
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
        else:
            virt_labelss = [[unique_label() for _ in virt_dimss[i]] for i in virt_dimss]
    else:
        virt_labelss = [v if type(v)==list else [v] for v in virt_labelss]
        if virt_dimss is None:
            virt_dimss = [(chi,)*len(virt_labels) for virt_labels in virt_labelss]
            
    bdts = []
    for bondsite in range(length):
        bdts.append( tni.random_diagonalTensor(virt_dimss[bondsite], virt_labelss[bondsite], dtype=dtype) )

    tensors = []
    for site in range(length):
        tensors.append( tni.random_tensor( virt_dimss[site]+phys_dimss[site]+virt_dimss[(site+1)%length], virt_labelss[site]+phys_labelss[site]+virt_labelss[(site+1)%length] , dtype=dtype) )

    return Cyc1DBTPS(tensors, bdts, phys_labelss)


