__all__ = ["Tensor"]

from tanuki.tnxp import xp as xp
import copy as copyModule
import warnings
from numpy import prod as soujou
import textwrap

class Tensor:
    def __init__(self, data, labels, copy=False):
        if not copy and isinstance(data, xp.ndarray):
            self.data = data
        else:
            self.data = xp.asarray(data)
        print(type(self.data))
        self.labels = list(labels)

    def copy(self):
        return Tensor(self.data, self.labels, copy=True)

    def __repr__(self):
        return f"Tensor(data={self.data}, labels={self.labels})"

    def __str__(self):
        if soujou(self.shape) > 64:
            dataStr = \
            "["*self.ndim + " ... " + "]"*self.ndim
        else:
            dataStr = str(self.data)
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"Tensor(\n" + \
        dataStr + "\n" + \
        f"    labels={self.labels},\n" + \
        f"    shape={self.shape},\n" + \
        f")"

        return re

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        if len(labels) != len(self.data.shape):
            raise ValueError(f"Labels do not match shape of data. labels=={labels}, shape=={self.data.shape}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"Labels are not unique. labels=={labels}")
        self._labels = list(labels)

    labels = property(get_labels, set_labels)

    