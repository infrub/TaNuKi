from tanuki.tnxp import xp as xp
from tanuki import tensor_core as tnc


def random_tensor(shape, labels=None, base_label=None, dtype=float):
    if dtype==float:
        data = xp.random.rand(*shape)*2-1
    elif dtype==complex:
        data = (xp.random.rand(*shape)*2-1) + 1j*(xp.random.rand(*shape)*2-1)
    else:
        raise ValueError("random_tensor argument dtype must be float or complex")
    return tnc.Tensor(data, labels=labels, base_label=base_label)


def zeros_tensor(shape, labels=None, base_label=None, dtype=float):
    data = xp.zeros(shape, dtype=dtype)
    return tnc.Tensor(data, labels=labels, base_label=base_label)


def identity_tensor(dim, labels=None, base_label=None, dtype=float):
    matrix = xp.ones(dim, dtype=dtype)
    if isinstance(labels, list) and len(labels)==1:
        labels = [labels[0], labels[0]]
    if not isinstance(labels, list) and not labels is None:
        labels = [labels, labels]
    tensor = tnc.DiagonalTensor(matrix, labels=labels, base_label=base_label)
    return tensor


def random_tensor_like(tensor):
    return random_tensor(tensor.shape, labels=tensor.labels, dtype=tensor.dtype)

def zeros_tensor_like(tensor):
    return zeros_tensor(tensor.shape, labels=tensor.labels, dtype=tensor.dtype)