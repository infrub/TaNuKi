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

def outofplacable(f):
    def g(self, *args, inplace=True, **kwargs):
        if inplace:
            f(self, *args, **kwargs)
        else:
            copied = self.copy(shallow=True)
            f(copied, *args, **kwargs)
            return copied
    return g


class Tensor:
    def __init__(self, data, labels, copy=False):
        if not copy and isinstance(data, xp.ndarray):
            self.data = data
        else:
            self.data = xp.asarray(data)
        self.labels = labels

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
    @outofplacable
    def move_index_to_top(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, 0)


    @outofplacable
    def move_index_to_bottom(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, self.ndim)

    @outofplacable
    def move_index_to_position(self, labelMove, position, inplace=True):
        indexMoveFrom = self.index_of_label(labelMove)
        if position == indexMoveFrom:
            return
        self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)
        if position < indexMoveFrom:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position)
        else:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position+1)

    @outofplacable
    def move_indices_to_top(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
        newIndicesMoveTo = list(range(len(oldIndicesMoveFrom)))

        oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
        #newIndicesNotMoveTo = list(range(len(oldIndicesMoveFrom), len(self.labels)))

        oldLabels = self.labels
        newLabels = [oldLabels[oldIndex] for oldIndex in oldIndicesMoveFrom] + [oldLabels[oldIndex] for oldIndex in oldIndicesNotMoveFrom]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable
    def move_indices_to_bottom(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
        newIndicesMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom), self.ndim))

        oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
        #newIndicesNotMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom)))

        oldLabels = self.labels
        newLabels = [oldLabels[oldIndex] for oldIndex in oldIndicesMoveFrom] + [oldLabels[oldIndex] for oldIndex in oldIndicesNotMoveFrom]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = [self.index_of_label(label) for label in labelsMove]
        newIndicesMoveTo = list(range(position, position+len(labelsMove)))

        oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
        newIndicesNotMoveTo = list(range(position)) + list(range(position+len(labelsMove), self.ndim))

        oldLabels = self.labels
        newLabels = [None]*len(oldLabels)
        for oldIndex, newIndex in zip(oldIndicesMoveFrom, newIndicesMoveTo):
            newLabels[newIndex] = oldLabels[oldIndex]
        for oldIndex, newIndex in zip(oldIndicesNotMoveFrom, newIndicesNotMoveTo):
            newLabels[newIndex] = oldLabels[oldIndex]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable
    def move_all_indices(self, newLabels):
        newLabels = normalize_argument_labels(newLabels)
        oldLabels = self.labels

        #oldPositions = list(range(self.ndim))
        newPositions = [newLabels.index(label) for label in oldLabels]

        self.data = xp.transpose(self.data, newPositions)
        self.labels = newLabels

