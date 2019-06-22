from tanuki.tnxp import xp as xp
from tanuki.utils import *
import copy as copyModule
import warnings
import textwrap
from collections import OrderedDict
import uuid
import random
from math import sqrt



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





class InputLengthError(Exception):
    pass

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



    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    #TODO pass if newIndices==oldIndices
    @inplacable_tensorMixin_method
    def move_index_to_top(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        newData = xp.rollaxis(self.data, indexMoveFrom, 0)
        newLabels = self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_index_to_bottom(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        newData = xp.rollaxis(self.data, indexMoveFrom, self.ndim)
        newLabels = self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:] + self.labels[indexMoveFrom:indexMoveFrom+1]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_index_to_position(self, indexMoveFrom, position):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)
        if position <= indexMoveFrom:
            newData = xp.rollaxis(self.data, indexMoveFrom, position)
            newLabels = self.labels[:position] + self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[position:indexMoveFrom] + self.labels[indexMoveFrom+1:]
        else:
            newData = xp.rollaxis(self.data, indexMoveFrom, position+1)
            newLabels = self.labels[:indexMoveFrom] + self.labels[indexMoveFrom+1:position+1] + self.labels[indexMoveFrom:indexMoveFrom+1] + self.labels[position+1:]
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_top(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(moveFrom) + self.labels_of_indices(notMoveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_bottom(self, moveFrom):
        moveFrom = self.normarg_indices_back(moveFrom)
        moveTo = list(range(self.ndim-len(moveFrom), self.ndim))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom) + self.labels_of_indices(moveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_indices_to_position(self, moveFrom, position):
        moveFrom = self.normarg_indices(moveFrom)
        moveTo = list(range(position, position+len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom)
        newLabels = newLabels[:position] + self.labels_of_indices(moveFrom) + newLabels[position:]
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def move_all_indices(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        if len(moveFrom) != self.ndim:
            raise InputLengthError()
        moveTo = list(range(self.ndim))
        newLabels = self.labels_of_indices(moveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)



    #methods for fuse/split
    #if new.. is no specified, assume like following:
    #["a","b","c","d"] <=split / fuse=> ["a",("b","c"),"d"]
    @outofplacable_tensorMixin_method
    def fuse_indices(self, splittedLabels=None, fusedLabel=None, memo=None):
        if memo is None:
            memo = {}

        if splittedLabels is None:
            if "splittedLabels" in memo:
                splittedLabels = memo["splittedLabels"]
            else:
                raise ValueError
        splittedIndices = self.normarg_indices(splittedLabels)
        splittedLabels = self.labels_of_indices(splittedIndices)

        if fusedLabel is None:
            if "fusedLabel" in memo:
                fusedLabel = memo["fusedLabel"]
            else:
                fusedLabel = tuple(splittedLabels)

        position = min(splittedIndices)
        self.move_indices_to_position(splittedIndices, position)
        del splittedIndices

        oldShape = self.shape
        splittedShape = oldShape[position:position+len(splittedLabels)]
        fusedDim = soujou(splittedShape)
        newShape = oldShape[:position] + (fusedDim,) + oldShape[position+len(splittedLabels):]

        oldLabels = self.labels
        newLabels = oldLabels[:position] + [fusedLabel] + oldLabels[position+len(splittedLabels):]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        memo.update({"splittedShape":splittedShape, "splittedLabels":splittedLabels, "fusedDim":fusedDim, "fusedLabel":fusedLabel})
        return memo #if out-of-place not returned. if you want, prepare a dict as memo in argument

    @outofplacable_tensorMixin_method
    def split_index(self, fusedLabel=None, splittedShape=None, splittedLabels=None, memo=None):
        if memo is None:
            memo = {}

        if fusedLabel is None:
            if "fusedLabel" in memo:
                fusedLabel = memo["fusedLabel"]
            else:
                raise ValueError
        fusedIndex = self.normarg_index(fusedLabel)
        fusedLabel = self.label_of_index(fusedIndex)

        if splittedShape is None:
            if "splittedShape" in memo:
                splittedShape = memo["splittedShape"]
            else:
                raise ValueError
        splittedShape = tuple(splittedShape)

        if splittedLabels is None:
            if "splittedLabels" in memo:
                splittedLabels = memo["splittedLabels"]
            else:
                splittedLabels = list(fusedLabel)
        splittedLabels = normarg_labels(splittedLabels)
        
        assert len(splittedLabels) == len(splittedShape)

        fusedDim = self.dim(fusedIndex)
        position = fusedIndex
        del fusedIndex

        assert soujou(splittedShape) == fusedDim

        newShape = self.shape[:position] + splittedShape + self.shape[position+1:]
        newLabels = self.labels[:position] + splittedLabels + self.labels[position+1:]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        memo.update({"fusedDim":fusedDim, "fusedLabel":fusedLabel, "splittedShape":splittedShape, "splittedLabels":splittedLabels})
        return memo #if out-of-place not returned. if you want, prepare a dict as memo in argument



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



    # MUST bag{moveFrom}==bag{0:self.ndim} (else: idk)
    # WILL moveFrom keep diagonality (else: CantKeepDiagonalityError)
    @inplacable_tensorMixin_method
    def move_all_indices_assuming_can_keep_diagonality(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        if len(moveFrom)!=self.ndim:
            raise InputLengthError()

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

    def move_all_indices(self, moveFrom):
        try:
            return self.move_all_indices_assuming_can_keep_diagonality(moveFrom)
        except CantKeepDiagonalityError:
            return self.to_tensor().move_all_indices(moveFrom)

    def move_half_all_indices_to_top(self, halfMoveFrom):
        halfMoveFrom = self.normarg_indices_front(halfMoveFrom)
        if len(halfMoveFrom) != self.halfndim:
            raise InputLengthError()
        moveFrom = halfMoveFrom + [(x+self.halfndim)%self.ndim for x in halfMoveFrom]
        if not eq_list(moveFrom, list(range(self.ndim))):
            raise CantKeepDiagonalityError()
        return self.move_all_indices(moveFrom)



    #methods for trace, contract
    # A[i,j,k,l,m,n] = [i==l][j==m][k==n]a[i,j,k]
    # \sum[j==k]A[i,j,k,l,m,n] = \sum_j [i==l][j==m][j==n]a[i,m,n] = [i==l][m==n] \sum_j a[i,m,n]
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

        newData = xp.diagonal(self.data, axis1=halfindex1, axis2=halfindex2)

        newLabels = self.labels[0:halfindex1]+self.labels[halfindex1+1:halfindex2]+self.labels[halfindex2+1:self.halfndim]
            + self.labels[coindex1:coindex1+1]
            + self.labels[self.halfndim:self.halfndim+halfindex1]+self.labels[self.halfndim+halfindex1+1:self.halfndim+halfindex2]+self.labels[self.halfndim+halfindex2+1:self.ndim]
            + self.labels[coindex2:coindex2+1]

        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def contract_internal_indices(self, indices1, indices2):
        indices1 = self.normarg_indices_front(indices1)
        indices2 = self.normarg_indices_front(indices2)

        temp = self.copy(shallow=True)
        while len(indices1)!=0:
            index1 = indices1.pop()
            index2 = indices2.pop()
            index1, index2 = min(index1,index2), max(index1,index2)
            coindex1, coindex2 = (index1+self.halfndim)%self.ndim, (index1+self.halfndim)%self.ndim
            halfindex1, halfindex2 = index1%self.halfndim, index2%self.halfndim
            halfindex1, halfindex2 = min(halfindex1, halfindex2), max(halfindex1, halfindex2)

            temp = temp.contract_internal_index(index1, index2)

            if index1+self.halfndim == index2:
                def dokoitta(x):
                    if 0<=x<index1: return x
                    elif index1<x<index2: return x-1
                    elif index2<x<self.ndim: return x-2
                    else: raise IndexError()
            else:
                def dokoitta(x):
                    if 0<=x<halfindex1: return x
                    elif halfindex1<x<halfindex2: return x-1
                    elif halfindex2<x<self.halfndim: return x-2
                    elif x==coindex1: return self.halfndim-2
                    elif self.halfndim<=x<self.halfndim+halfindex1: return x-1
                    elif self.halfndim+halfindex1<x<self.halfndim+halfindex2: return x-2
                    elif self.halfndim+halfindex2<x<self.ndim: return x-3
                    elif x==coindex2: return self.ndim-3
                    else: raise IndexError()

            indices1 = [dokoitta(x) for x in indices1]
            indices2 = [dokoitta(x) for x in indices2]

        return temp

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
        elif type(index1) == list:
            return self.contract_internal_indices(index1, index2)
        else:
            return self.contract_internal_index(index1, index2)

    trace = contract_internal
    tr = contract_internal





#contract functions
class ToContract:
    #A["a"]*B["b"] == contract(A,B,["a"],["b"])
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

    def __mul__(self, other):
        return contract(self.tensor, other.tensor, self.labels, other.labels)



def direct_product(A, B):
    if type(A)==Tensor and type(B)==Tensor:
        cData = xp.tensordot(A.data, B.data, 0)
        cLabels = A.labels + B.labels
        return Tensor(cData, cLabels)
    elif type(A)==DiagonalTensor and type(B)==DiagonalTensor:
        cData = xp.tensordot(A.data, B.data, 0)
        cLabels = A.labels[:A.halfndim] + B.labels[:B.halfndim] + A.labels[A.halfndim:] + B.labels[B.halfndim:]
        return DiagonalTensor(cData, cLabels)




def contract(A, B, aIndicesContract, bIndicesContract):
    aIndicesContract = A.normarg_indices_back(aIndicesContract)
    bIndicesContract = B.normarg_indices_front(bIndicesContract)
    aLabelsContract = A.labels_of_indices(aIndicesContract)
    bLabelsContract = B.labels_of_indices(bIndicesContract)
    aDimsContract = A.dims(aIndicesContract)
    bDimsContract = A.dims(bIndicesContract)
    assert aDimsContract == bDimsContract: f"{A}, {B}, {aLabelsContract}, {bLabelsContract}"

    if type(A)==Tensor and type(B)==Tensor:
        cData = xp.tensordot(A.data, B.data, (aIndicesContract, bIndicesContract))
        cLabels = more_popped_list(A.labels, aIndicesContract) + more_popped_list(B.labels, bIndicesContract)
        return Tensor(cData, cLabels)

    elif type(A)==DiagonalTensor and type(B)==DiagonalTensor:
        try:
            A = A.move_half_all_indices_to_top(aIndicesContract)
            B = B.move_half_all_indices_to_top(bIndicesContract)
            cData = A.data * B.data
            cLabels = A.labels[A.halfndim:] + B.labels[B.halfndim:]
            return DiagonalTensor(cData, cLabels)
        except (InputLengthError, CantKeepDiagonalityError):
            C = direct_product(A, B)
            cIndicesContract1 = [x if x<A.halfndim else x+B.halfndim for x in aIndicesContract]
            cIndicesContract2 = [x+A.halfndim if x<B.halfndim else x+A.ndim for x in aIndicesContract]
            return C.contract_internal_indices(cIndicesContract1, cIndicesContract2)

    elif type(A)==Tensor and type(B)==DiagonalTensor:
        try:
            B = B.move_half_all_indices_to_top(bIndicesContract)
            A = A.move_indices_to_bottom(aIndicesContract)
            aIndicesNotContract = diff_list(list(range(A.ndim)), aIndicesContract)
            bIndicesNotContract = diff_list(list(range(B.ndim)), bIndicesContract)
            aLabelsNotContract = A.labels_of_indices(aIndicesNotContract)
            bLabelsNotContract = B.labels_of_indices(bIndicesNotContract)
            aDimsNotContract = A.dims(aIndicesNotContract)
            bDimsNotContract = B.dims(bIndicesNotContract)
            a = A.data.reshape((soujou(aDimsNotContract), soujou(aDimsContract)))
            b = B.data.flatten()
            c = xp.multiply(a,b)
            c = c.reshape(aDimsNotContract+bDimsNotContract)
            cLabels = aLabelsNotContract+bLabelsNotContract
            return Tensor(c, cLabels)
        except InputLengthError, CantKeepDiagonalityError:
            return contract(A, diagonalTensor_to_tensor(B), aIndicesContract, bIndicesContract)

    elif type(A)==DiagonalTensor and type(B)==Tensor:
        return contract(B, A, bIndicesContract, aIndicesContract)

    else:
        return NotImplemented










#converting functions
def tensor_to_ndarray(T, indices):
    T = T.move_all_indices(indices)
    return T.data

def tensor_to_matrix(T, rows, cols=None):
    rows, cols = T.normarg_complement_indices(rows, cols)
    T = T.move_all_indices(rows+cols)
    total_row_dim = soujou(T.shape[:len(rows)])
    total_col_dim = soujou(T.shape[len(rows):])
    return xp.reshape(t.data, (total_row_dim, total_col_dim))

def tensor_to_vector(T, indices):
    T = T.move_all_indices(indices)
    return xp.reshape(T.data, (T.size,))

def tensor_to_scalar(T):
    return xp.asscalar(T.data)



def ndarray_to_tensor(ndarray, labels):
    return Tensor(ndarray, labels)

def matrix_to_tensor(matrix, shape, labels):
    return Tensor(xp.reshape(matrix, shape), labels)

def vector_to_tensor(vector, shape, labels):
    return Tensor(xp.reshape(vector, shape), labels)

def scalar_to_tensor(scalar):
    return Tensor(scalar, [])



def diagonalTensor_to_diagonalElementsNdarray(DT, indices):
    DT = DT.move_all_indices_assuming_can_keep_diagonality(indices)
    return DT.data

def diagonalTensor_to_diagonalElementsVector(DT, indices):
    DT = DT.move_all_indices_assuming_can_keep_diagonality(indices)
    return xp.flatten(DT.data)



def diagonalElementsNdarray_to_diagonalTensor(ndarray, labels):
    return DiagonalTensor(ndarray, labels)

def diagonalElementsVector_to_diagonalTensor(vector, halfshape, labels):
    return diagonalElementsNdarray_to_diagonalTensor(vector.reshape(halfshape), labels)



def diagonalTensor_to_tensor(DT):
    shape = DT.shape
    return Tensor(xp.diagflat(diagonalTensor.data).reshape(shape), diagonalTensor.labels)

def diagonalTensor_to_matrix(DT, rows, cols=None):
    return tensor_to_matrix(diagonalTensor_to_tensor(DT), rows, cols)

def diagonalTensor_to_vector(DT, indices):
    return tensor_to_vector(diagonalTensor_to_tensor(DT), indices)

def diagonalTensor_to_scalar(DT):
    return xp.asscalar(DT.data)



def tensor_to_diagonalTensor(T, indices):
    T = T.move_all_indices(indices)
    halfshape = T.shape[:T.ndim//2]
    halfsize = soujou(halfshape)
    t = T.data
    t = t.reshape((halfsize, halfsize))
    dt = xp.diagonal(t)
    dt = dt.reshape(halfshape)
    return DiagonalTensor(dt, T.labels)

def matrix_to_diagonalTensor(matrix, halfshape, labels):
    dt = xp.diagonal(matrix)
    dt = dt.reshape(halfshape)
    return DiagonalTensor(dt, labels)

def vector_to_diagonalTensor(vector, halfshape, labels):
    halfsize = soujou(halfshape)
    matrix = vector.reshape((halfsize, halfsize))
    dt = xp.diagonal(matrix)
    dt = dt.reshape(halfshape)
    return DiagonalTensor(dt, labels)

def scalar_to_diagonalTensor(scalar):
    return DiagonalTensor(scalar, [])

