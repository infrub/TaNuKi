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





class CantKeepDiagonalityError(Exception):
    pass





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



    def __getitem__(self, *indices):
        if len(indices)==1 and isinstance(indices[0],list): #if called like as A[["a"]]
            indices = indices[0]
        else:
            indices = list(indices)
        return ToContract(self, indices)





class Tensor(TensorMixin):
    #basic methods
    def __init__(self, data, labels=None, base_label=None, copy=False):
        if not copy and isinstance(data, xp.ndarray):
            self.data = data
        else:
            self.data = xp.asarray(data)
        if labels is None:
            if base_label is None:
                base_label = unique_label()
            self.assign_labels(base_label)
        else:
            self.labels = labels

    def copy(self, shallow=False):
        return Tensor(self.data, self.labels, copy=not(shallow))

    def __repr__(self):
        return f"Tensor(data={self.data}, labels={self.labels})"

    def __str__(self):
        if self.size > 100:
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



    #properties
    @property
    def shape(self): #tuple
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype



    @inplacable_tensorMixin_method
    def arrange_indices(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        assert len(moveFrom) == self.ndim
        moveTo = list(range(self.ndim))
        newLabels = self.labels_of_indices(moveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)



    #methods for trace, contract
    @inplacable_tensorMixin_method
    def contract_internal_index(self, index1, index2):
        index1 = self.normarg_index_front(index1)
        index2 = self.normarg_index_back(index2)
        index1, index2 = min(index1,index2), max(index1,index2)

        newData = xp.trace(self.data, axis1=index1, axis2=index2)
        newLabels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]

        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def contract_internal_common(self):
        temp = self
        commons = floor_half_list(temp.labels)
        for common in commons:
            temp = temp.contract_internal(common, common)
        return temp

    @inplacable_tensorMixin_method
    def contract_internal(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_common_internal()
        else:
            return self.contract_internal(index1, index2)

    trace = contract_internal
    tr = trace








# A[i,j,k,l] = [i==k][j==l]A.data[i,j]
class DiagonalTensor(TensorMixin):
    #basic methods
    def __init__(self, data, labels=None, base_label=None, copy=False):
        if not copy and isinstance(data, xp.ndarray):
            self.data = data
        else:
            self.data = xp.asarray(data)
        if labels is None:
            if base_label is None:
                base_label = unique_label()
            self.assign_labels(base_label)
        else:
            self.labels = labels

    def copy(self, shallow=False):
        return DiagonalTensor(self.data, self.labels, copy=not(shallow))

    def __repr__(self):
        return f"DiagonalTensor(data={self.data}, labels={self.labels})"

    def __str__(self):
        if self.halfsize > 100:
            dataStr = \
            "["*self.halfndim + " ... " + "]"*self.halfndim
        else:
            dataStr = str(self.data)
        dataStr = textwrap.indent(dataStr, "    ")

        re = \
        f"DiagonalTensor(\n" + \
        dataStr + "\n" + \
        f"    labels={self.labels},\n" + \
        f"    shape={self.shape},\n" + \
        f")"

        return re



    #properties
    @property
    def halfshape(self): #tuple
        return self.data.shape
    
    @property
    def halfndim(self):
        return self.data.ndim

    @property
    def halfsize(self):
        return self.data.size
    
    @property
    def halfshape(self):
        return self.halfshape + self.halfshape
    
    @property
    def halfndim(self):
        return self.halfndim * 2

    @property
    def halfsize(self):
        return self.halfsize * self.halfsize

    @property
    def dtype(self):
        return self.data.dtype



    @inplacable_tensorMixin_method
    def arrange_indices_assuming_can_keep_diagonality(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)

        data = self.data
        labels = copyModule.copy(self.labels)

        for x in range(self.halfndim):
            i,j = moveFrom[x], moveFrom[self.halfndim+x]
            if i+self.halfndim==j:
                continue
            elif j+self.halfndim==i:
                labels[i], labels[j] = labels[j], labels[i]
                moveFrom[x], moveFrom[self.halfndim+x] = j, i
            else:
                raise CantKeepDiagonalityError()

        halfMoveFrom = moveFrom[:self.halfndim]
        halfMoveTo = list(range(self.halfndim))
        data = xp.moveaxis(data, halfMoveFrom, halfMoveTo)
        labels = [labels[i] for i in moveFrom]

        return DiagonalTensor(data, labels)

    def arrange_indices(self, moveFrom):
        try:
            return self.arrange_indices_assuming_can_keep_diagonality(moveFrom)
        except CantKeepDiagonalityError as e:
            return self.to_tensor().arrange_indices(moveFrom)



    #methods for trace, contract
    # A[i,j,k,l,m,n] = [i==l][j==m][k==n]a[i,j,k]
    # \sum[j==k]A[i,j,k,l,m,n] = \sum_j [i==l][j==m][j==n]a[i,j,j] = [i==l][m==n] \sum_j a[i,j,j]
    # TODO not tested
    @inplacable_tensorMixin_method
    def contract_internal_index(self, index1, index2):
        index1 = self.normarg_index_front(index1)
        index2 = self.normarg_index_back(index2)
        index1, index2 = min(index1,index2), max(index1,index2)

        if index1+self.halfndim == index2:
            newData = xp.sum(self.data, axis=index1)
            newLabels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]
            return Tensor(newData, newLabels)

        coindex1, coindex2 = (index1+self.halfndim)%self.ndim, (index1+self.halfndim)%self.ndim
        halfindex1, halfindex2 = index1%self.halfndim, index2%self.halfndim
        halfindex1, halfindex2 = min(halfindex1, halfindex2), max(halfindex1, halfindex2)

        newData = xp.trace(self.data, axis1=halfindex1, axis2=halfindex2)
        newData = xp.tile(newData, (self.dim(coindex1),)+(1,)*(self.halfndim-2)) #TODO tadasii?

        newLabels = self.labels[coindex1:coindex1+1]
                    +self.labels[0:halfindex1]+self.labels[halfindex1+1:halfindex2]+self.labels[halfindex2+1:self.halfndim]
                    +self.labels[coindex2:coindex2+1]
                    +self.labels[self.halfndim:self.halfndim+halfindex1]+self.labels[self.halfndim+halfindex1+1:self.halfndim+halfindex2]+self.labels[self.halfndim+halfindex2+1:self.ndim]

        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def contract_internal_common(self):
        temp = self
        commons = floor_half_list(temp.labels)
        for common in commons:
            temp = temp.contract_internal(common, common)
        return temp

    def contract_internal(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_internal_common()
        else:
            return self.contract_internal_index(index1, index2)

    trace = contract_internal
    tr = contract_internal