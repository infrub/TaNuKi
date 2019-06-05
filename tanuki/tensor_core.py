__all__ = ["Tensor"]

from tanuki.tnxp import xp as xp
import copy as copyModule
import warnings
from numpy import prod as soujou
import textwrap


def normalize_argument_labels(labels):
    if isinstance(labels, list):
        return labels
    elif isinstance(labels, tuple):
        return list(labels)
    else:
        return [labels]



class Tensor:
    def __init__(self, data, labels, copy=False):
        if not copy and isinstance(data, xp.ndarray):
            self.data = data
        else:
            self.data = xp.asarray(data)
        self.labels = list(labels)

    def copy(self, shallow=False):
        return Tensor(self.data, self.labels, copy=not(shallow))

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
            raise ValueError(f"labels do not match shape of data. labels=={labels}, shape=={self.data.shape}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"labels are not unique. labels=={labels}")
        self._labels = list(labels)

    labels = property(get_labels, set_labels)


    def index_of_label(self, label):
        return self.labels.index(label)

    def dim_of_index(self, index):
        return self.data.shape[index]

    def dim_of_label(self, label):
        return self.dim_of_index(self.index_of_label(label))


    def replace_label(self, oldLabels, newLabels):
        oldLabels = normalize_argument_labels(oldLabels)
        newLabels = normalize_argument_labels(newLabels)
        for i, label in enumerate(self.labels):
            if label in oldLabels:
                self.labels[i] = newLabels[oldLabels.index(label)]


    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    def move_index_to_top(self, labelMove, inplace=True):
        if inplace:
            indexMoveFrom = self.index_of_label(labelMove)
            self.labels.pop(indexMoveFrom)
            self.labels.insert(0, labelMove)
            self.data = xp.rollaxis(self.data, indexMoveFrom, 0)
        else:
            copySelf = self.copy(shallow=True)
            copySelf.move_index_to_top(labelMove, inplace=True)
            return copySelf

    def move_index_to_bottom(self, labelMove, inplace=True):
        if inplace:
            indexMoveFrom = self.index_of_label(labelMove)
            self.labels.pop(indexMoveFrom)
            self.labels.append(labelMove)
            self.data = xp.rollaxis(self.data, indexMoveFrom, self.ndim)
        else:
            copySelf = self.copy(shallow=True)
            copySelf.move_index_to_bottom(labelMove, inplace=True)
            return copySelf

    def move_index_to_position(self, labelMove, position, inplace=True):
        if inplace:
            indexMoveFrom = self.index_of_label(labelMove)
            if position == indexMoveFrom:
                return
            self.labels.pop(indexMoveFrom)
            self.labels.insert(position, labelMove)
            if position < indexMoveFrom:
                self.data = xp.rollaxis(self.data, indexMoveFrom, position)
            else:
                self.data = xp.rollaxis(self.data, indexMoveFrom, position+1)
        else:
            copySelf = self.copy(shallow=True)
            copySelf.move_index_to_position(labelMove, position, inplace=True)
            return copySelf
