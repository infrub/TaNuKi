from tanuki.tnxp import xp as xp
from tanuki.utils import *
import copy as copyModule
import warnings
import textwrap
from collections import OrderedDict
import uuid
import random



#decorators
#decorate in-place Tensor class method with @outofplacable_tensorMixin_method to be able to use as out-of-place method.
def outofplacable_tensorMixin_method(f):
    def g(self, *args, inplace=True, **kwargs):
        if inplace:
            return f(self, *args, **kwargs)
        else:
            copied = self.copy(shallow=True)
            f(copied, *args, **kwargs)
            return copied
    return g

#decorate out-of-place Tensor class method with @inplacable_tensorMixin_method to be able to use as out-of-place method.
def inplacable_tensorMixin_method(f):
    def g(self, *args, inplace=False, **kwargs):
        if inplace:
            re = f(self, *args, **kwargs)
            self.data = re.data
            self.labels = re.labels
        else:
            return f(self, *args, **kwargs)
    return g



#classes
class TensorMixin:
    def __copy__(self):
        return self.copy(shallow=True)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        return self.copy(shallow=False)



    #methods for labels
    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        assert len(labels) == self.ndim, f"{labels}, {self.shape}"
        self._labels = list(labels)

    labels = property(get_labels, set_labels)



    def label_of_index(self, index):
        return self.labels[index]

    def labels_of_indices(self, indices):
        return [self.labels[index] for index in indices]



    def index_of_label_front(self, label):
        return self.labels.index(label)

    def index_of_label_back(self, label):
        return self.ndim - 1 - list(reversed(self.labels)).index(label)

    index_of_label = index_of_label_front

    def indices_of_labels_front(self, labels): #list[int]
        return indexs_duplable_front(self.labels, labels)

    def indices_of_labels_back(self, labels): #list[int]
        return indexs_duplable_back(self.labels, labels)

    indices_of_labels = indices_of_labels_front



    def normarg_index_front(self, indexOrLabel):
        if type(indexOrLabel)==int:
            return indexOrLabel
        elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
            return self.index_of_label_front(indexOrLabel)
        else:
            raise TypeError

    def normarg_index_back(self, indexOrLabel):
        if type(indexOrLabel)==int:
            return indexOrLabel
        elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
            return self.index_of_label_back(indexOrLabel)
        else:
            raise TypeError

    normarg_index = normarg_index_front

    def normarg_indices_front(self, indicesOrLabels):
        if type(indicesOrLabels)!=list:
            indicesOrLabels = [indicesOrLabels]
        temp = list(self.labels)
        res = []
        for indexOrLabel in indicesOrLabels:
            if type(indexOrLabel)==int:
                index = indexOrLabel
            elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
                index = temp.index(indexOrLabel)
            else:
                raise TypeError(f"indicesOrLabels=={indicesOrLabels}, type(indicesOrLabels)=={type(indicesOrLabels)}")
            res.append(index)
            temp[index] = None
        return res

    def normarg_indices_back(self, indicesOrLabels):
        if type(indicesOrLabels)!=list:
            indicesOrLabels = [indicesOrLabels]
        indicesOrLabels = reversed(indicesOrLabels)
        temp = list(reversed(self.labels))
        res = []
        for indexOrLabel in indicesOrLabels:
            if type(indexOrLabel)==int:
                index = indexOrLabel
                revindex = len(temp)-1-index
            elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
                revindex = temp.index(indexOrLabel)
                index = len(temp)-1-revindex
            else:
                raise TypeError(f"indicesOrLabels=={indicesOrLabels}, type(indicesOrLabels)=={type(indicesOrLabels)}")
            res.append(index)
            temp[revindex] = None
        return list(reversed(res))

    normarg_indices = normarg_indices_front

    def normarg_complement_indices(self, row_indices, column_indices=None):
        row_indices = self.normarg_indices_front(row_indices)
        if column_indices is None:
            column_indices = diff_list(list(range(self.ndim)), row_indices)
        else:
            column_indices = self.normarg_indices_back(column_indices)
        return row_indices, column_indices



    def dim(self, index):
        index = self.normarg_index(index)
        return self.shape[index]

    def dims_front(self, indices):
        indices = self.normarg_indices_front(indices)
        return tuple(self.shape[index] for index in indices)

    def dims_back(self, indices):
        indices = self.normarg_indices_back(indices)
        return tuple(self.shape[index] for index in indices)

    dims = dims_front



    @outofplacable_tensorMixin_method
    def replace_labels(self, oldIndices, newLabels):
        oldIndices = self.normarg_indices(oldIndices)
        newLabels = normarg_labels(newLabels)
        for oldIndex, newLabel in zip(oldIndices, newLabels):
            self.labels[oldIndex] = newLabel

    @outofplacable_tensorMixin_method
    def aster_labels(self, oldIndices=None):
        if oldIndices is None: oldIndices=self.labels
        oldIndices = self.normarg_indices_front(oldIndices)
        oldLabels = self.labels_of_indices(oldIndices)
        newLabels = aster_labels(oldLabels)
        self.replace_labels(oldIndices, newLabels)

    @outofplacable_tensorMixin_method
    def unaster_labels(self, oldIndices=None):
        if oldIndices is None: oldIndices=self.labels
        oldIndices = self.normarg_indices_front(oldIndices)
        oldLabels = self.labels_of_indices(oldIndices)
        newLabels = unaster_labels(oldLabels)
        self.replace_labels(oldIndices, newLabels)

    def assign_labels(self, base_label):
        self.labels = [base_label+"_"+str(i) for i in range(self.ndim)]


