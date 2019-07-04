"""
Flexible numpy/cupy switcher.
If you wanna use cupy, tnxp.xp.use_cupy() after importing tanuki.

required methods:
    numpy:
        asarray, array, rollaxis, moveaxis, reshape, pad, trace, conj, tensordot, diag, sqrt, concatenate, dot, newaxis, zeros, identity, arange
    scipy.linalg:
        norm, svd, qr, eigh, LinAlgError
    scipy.sparse.linalg:
        eigsh, LinearOperator
    numpy.random:
        rand
"""


class Xp():
    def __init__(self):
        self.use_scipy()

    def use_scipy(self):
        self.isScipy = True
        self.isCupy = False
        import scipy
        import scipy.linalg
        import scipy.sparse.linalg
        import numpy.random
        self._core = scipy
        self.linalg = scipy.linalg
        self.sparse.linalg = scipy.sparse.linalg
        self.random = numpy.random

    def use_cupy(self):
        self.isCupy = True
        self.isScipy = False
        import cupy
        import cupy.linalg
        import cupy.random
        self._core = cupy
        self.linalg = cupy.linalg
        self.random = cupy.random

    def __getattr__(self, name):
        return self._core.__getattribute__(name)

xp = Xp()
