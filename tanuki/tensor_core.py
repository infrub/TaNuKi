from tanuki.tnxp import xp as xp
from tanuki.utils import *
import copy as copyModule
import warnings
from numpy import prod as soujou
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
    #methods for labels
    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        if len(labels) != len(self.shape):
            raise ValueError(f"labels do not match shape of data. labels=={labels}, shape=={self.shape}")
        #if len(labels) != len(set(labels)):
        #    warnings.warn(f"labels are not unique. labels=={labels}")
        self._labels = list(labels)

    labels = property(get_labels, set_labels)

    @outofplacable_tensorMixin_method
    def replace_labels(self, oldLabels, newLabels):
        #if tensor has labels with same name, all labels with the name is replaced. 
        oldLabels = normalize_argument_labels(oldLabels)
        tempLabelBase = unique_label()
        tempLabels = ["temp_replace_labels_"+tempLabelBase+"_"+str(i) for i in range(len(oldLabels))]
        newLabels = normalize_argument_labels(newLabels)
        for i, label in enumerate(self.labels):
            if label in oldLabels:
                self.labels[i] = tempLabels[oldLabels.index(label)]
        for i, label in enumerate(self.labels):
            if label in tempLabels:
                self.labels[i] = newLabels[tempLabels.index(label)]

    def assign_labels(self, base_label):
        self.labels = [base_label+"_"+str(i) for i in range(self.ndim)]


    def index_of_label(self, label):
        return self.labels.index(label)

    def dim_of_index(self, index):
        return self.shape[index]

    def dim_of_label(self, label):
        return self.dim_of_index(self.index_of_label(label))

    def indices_of_labels_front(self,labels): #list[int]
        return indexs_duplable_front(self.labels, labels)
    def indices_of_labels_back(self,labels): #list[int]
        return indexs_duplable_back(self.labels, labels)
    indices_of_labels = indices_of_labels_front

    def dims_of_indices(self,indices): #tuple[int]
        return tuple(self.dim_of_index(index) for index in indices)

    def dims_of_labels_front(self,labels): #tuple[int]
        return self.dims_of_indices(self.indices_of_labels_front(labels))
    def dims_of_labels_back(self,labels): #tuple[int]
        return self.dims_of_indices(self.indices_of_labels_back(labels))
    dims_of_labels = dims_of_labels_front

    #methods for basic operations
    @inplacable_tensorMixin_method
    def adjoint(self,row_labels,column_labels=None):
        row_labels, column_labels = normalize_and_complement_argument_labels(self.labels,row_labels,column_labels)
        if len(row_labels) != len(column_labels):
            raise ValueError(f"adjoint arg must be len(row_labels)==len(column_labels). but row_labels=={row_labels}, column_labels=={column_labels}")
        out = self.conjugate()
        out.replace_labels(row_labels+column_labels, column_labels+row_labels)
        return out

    adj = adjoint

    @inplacable_tensorMixin_method
    def hermite(self, row_labels, column_labels=None):
        return (self + self.adjoint(row_labels,column_labels))/2

    @inplacable_tensorMixin_method
    def antihermite(self, row_labels, column_labels=None):
        return (self - self.adjoint(row_labels,column_labels))/2





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
    @outofplacable_tensorMixin_method
    def move_index_to_top(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, 0)

    @outofplacable_tensorMixin_method
    def move_index_to_bottom(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, self.ndim)

    @outofplacable_tensorMixin_method
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

    @outofplacable_tensorMixin_method
    def move_indices_to_top(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
        newIndicesMoveTo = list(range(len(oldIndicesMoveFrom)))

        oldIndicesNotMoveFrom = diff_list(range(len(self.labels)), oldIndicesMoveFrom)
        #newIndicesNotMoveTo = list(range(len(oldIndicesMoveFrom), len(self.labels)))

        oldLabels = self.labels
        newLabels = [oldLabels[oldIndex] for oldIndex in oldIndicesMoveFrom] + [oldLabels[oldIndex] for oldIndex in oldIndicesNotMoveFrom]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
        newIndicesMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom), self.ndim))

        oldIndicesNotMoveFrom = diff_list(range(len(self.labels)), oldIndicesMoveFrom)
        #newIndicesNotMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom)))

        oldLabels = self.labels
        newLabels = [oldLabels[oldIndex] for oldIndex in oldIndicesNotMoveFrom] + [oldLabels[oldIndex] for oldIndex in oldIndicesMoveFrom]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
        newIndicesMoveTo = list(range(position, position+len(labelsMove)))

        oldIndicesNotMoveFrom = diff_list(range(len(self.labels)), oldIndicesMoveFrom)
        newIndicesNotMoveTo = list(range(position)) + list(range(position+len(labelsMove), self.ndim))

        oldLabels = self.labels
        newLabels = [None]*len(oldLabels)
        for oldIndex, newIndex in zip(oldIndicesMoveFrom, newIndicesMoveTo):
            newLabels[newIndex] = oldLabels[oldIndex]
        for oldIndex, newIndex in zip(oldIndicesNotMoveFrom, newIndicesNotMoveTo):
            newLabels[newIndex] = oldLabels[oldIndex]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_all_indices(self, newLabels):
        newLabels = normalize_argument_labels(newLabels)
        oldLabels = self.labels

        if sorted(newLabels) != sorted(oldLabels):
            raise ValueError(f"newLabels do not match oldLabels. oldLabels=={oldLabels}, newLabels=={newLabels}")

        #oldPositions = list(range(self.ndim))
        newPositions = indexs_duplable_front(newLabels, oldLabels)

        self.data = xp.transpose(self.data, newPositions)
        self.labels = newLabels


    #methods for fuse/split
    #if new.. is no specified, assume like following:
    #["a","b","c","d"] <=split / fuse=> ["a",("b","c"),"d"]
    @outofplacable_tensorMixin_method
    def fuse_indices(self, labelsFuse, newLabelFuse=None):
        labelsFuse = normalize_argument_labels(labelsFuse)
        if newLabelFuse is None:
            newLabelFuse = tuple(labelsFuse)

        position = min(self.indices_of_labels(labelsFuse))
        self.move_indices_to_position(labelsFuse, position)

        oldShape = self.shape
        oldShapeFuse = oldShape[position:position+len(labelsFuse)]
        newDimFuse = soujou(oldShapeFuse)
        newShape = oldShape[:position] + (newDimFuse,) + oldShape[position+len(labelsFuse):]

        oldLabels = self.labels
        newLabels = oldLabels[:position] + [newLabelFuse] + oldLabels[position+len(labelsFuse):]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        return OrderedDict(oldShapeFuse=oldShapeFuse, oldLabelsFuse=labelsFuse, labelsFuse=labelsFuse, newDimFuse=newDimFuse, newLabelFuse=newLabelFuse) #useful info (this is missed if out-of-place)

    @outofplacable_tensorMixin_method
    def split_index(self, labelSplit, newShapeSplit, newLabelsSplit=None):
        newShapeSplit = tuple(newShapeSplit)
        if newLabelsSplit is None:
            if isinstance(labelSplit, tuple):
                newLabelsSplit = list(labelSplit)
            else:
                newLabelsSplit = [labelSplit]
            if len(newLabelsSplit) != len(newShapeSplit):
                raise ValueError(f"newLabelsSplit is not defined and could not be assumed. labelSplit=={labelSplit}, newShapeSplit=={newShapeSplit}")
        else:
            newLabelsSplit = normalize_argument_labels(newLabelsSplit)

        indexSplit = self.index_of_label(labelSplit)
        newShape = self.shape[:indexSplit] + newShapeSplit + self.shape[indexSplit+1:]
        newLabels = self.labels[:indexSplit] + newLabelsSplit + self.labels[indexSplit+1:]

        self.data = xp.reshape(self.data, newShape)
        self.labels = newLabels

        return OrderedDict(oldDimSplit=soujou(newShapeSplit), oldLabelSplit=labelSplit, labelSplit=labelSplit, newShapeSplit=newShapeSplit, newLabelsSplit=newLabelsSplit)


    #methods for basic operations
    def __mul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(self, other)
        elif xp.isscalar(other):
            return Tensor(self.data*other, labels=self.labels)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(other, self)
        elif xp.isscalar(other):
            return Tensor(self.data*other, labels=self.labels)
        return NotImplemented

    def __truediv__(self, other):
        if xp.isscalar(other):
            return Tensor(self.data/other, labels=self.labels)
        return NotImplemented

    def __add__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            if type(other)==DiagonalTensor:
                other = other.to_tensor()
            return Tensor(self.data+other.data, self.labels)
        return NotImplemented

    def __radd__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            if type(other)==DiagonalTensor:
                other = other.to_tensor()
            return Tensor(other.data+self.data, self.labels)
        return NotImplemented

    def __sub__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            if type(other)==DiagonalTensor:
                other = other.to_tensor()
            return Tensor(self.data-other.data, self.labels)
        return NotImplemented

    def __rsub__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            if type(other)==DiagonalTensor:
                other = other.to_tensor()
            return Tensor(other.data-self.data, self.labels)
        return NotImplemented

    def __eq__(self, other, skipLabelSort=False, absolute_threshold=1e-10):
        if isinstance(other, TensorMixin):
            diff = self.__sub__(other, skipLabelSort=skipLabelSort)
            return diff.norm() <= absolute_threshold
        return NotImplemented


    @inplacable_tensorMixin_method
    def conjugate(self):
        return Tensor(data=self.data.conj(),labels=self.labels)

    conj = conjugate

    @inplacable_tensorMixin_method
    def pad_indices(self, labels, npads):
        indices = self.indices_of_labels(labels)
        wholeNpad = [(0,0)] * self.ndim
        for index,npad in zip(indices, npads):
            wholeNpad[index] = npad
        newData = xp.pad(self.data, wholeNpad, mode="constant", constant_values=0)
        return Tensor(newData, labels=self.labels)

    def norm(self): #Frobenius norm
        return xp.linalg.norm(self.data)

    @inplacable_tensorMixin_method
    def normalize(self):
        norm = self.norm()
        return self / norm


    #methods for trace, contract
    @inplacable_tensorMixin_method
    def contract_internal(self, label1, label2):
        index1 = indexs_duplable_front(self.labels, label1)
        index2 = indexs_duplable_back(self.labels, label2)
        index1, index2 = min(index1,index2), max(index1,index2)

        newData = xp.trace(self.data, axis1=index1, axis2=index2)
        newLabels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]

        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def contract_common_internal(self):
        temp = self
        commons = floor_half_list(temp.labels)
        for common in commons:
            temp = temp.contract_internal(common, common)
        return temp

    @inplacable_tensorMixin_method
    def trace(self, label1=None, label2=None):
        if label1 is None:
            return self.contract_common_internal()
        else:
            return self.contract_internal(label1, label2)

    tr = trace


    @inplacable_tensorMixin_method
    def contract(self, *args, **kwargs):
        return contract(self, *args, **kwargs)

    def __getitem__(self, *labels):
        if len(labels)==1 and isinstance(labels[0],list): #if called like as A[["a"]]
            labels = labels[0]
        else:
            labels = list(labels)
        return ToContract(self, labels)


    #methods for dummy index
    @outofplacable_tensorMixin_method
    def add_dummy_index(self, label=()):
        self.data = self.data[xp.newaxis, :]
        self.labels.insert(0, label)

    @outofplacable_tensorMixin_method
    def remove_all_dummy_indices(self, labels=None):
        oldShape = self.shape
        oldLabels = self.labels
        newShape = ()
        newLabels = []
        for i, x in enumerate(oldLabels):
            if oldShape[i]==1 and ((labels is None) or (x in labels)):
                pass
            else:
                newShape = newShape + (oldShape[i],)
                newLabels.append(x)
        self.data = self.data.reshape(newShape)
        self.labels = newLabels


    #methods for converting to simple linalg object
    def to_diagonalTensor(self):
        return tensor_to_diagonalTensor(self)

    def to_matrix(self, row_labels, column_labels=None):
        return tensor_to_matrix(self, row_labels, column_labels)

    def to_vector(self, labels):
        return tensor_to_vector(self, labels)

    def to_scalar(self):
        return tensor_to_scalar(self)


    #methods for confirming character
    def is_scalar(self):
        return self.ndim==0

    def is_diagonal(self, absolute_threshold=1e-10):
        if self.ndim != 2:
            return False
        temp = self.data[xp.eye(*self.data.shape)==0]
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_identity(self, absolute_threshold=1e-10):
        if self.ndim != 2:
            return False
        temp = self.data - xp.eye(*self.data.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_right_unitary(self, column_labels, absolute_threshold=1e-10):
        column_labels, row_labels = normalize_and_complement_argument_labels(self.labels, column_labels)
        M = self.to_matrix(row_labels, column_labels)
        temp = xp.dot(M, M.conj().transpose())
        temp = temp - xp.eye(*temp.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_left_unitary(self, row_labels, absolute_threshold=1e-10):
        row_labels, column_labels = normalize_and_complement_argument_labels(self.labels, row_labels)
        M = self.to_matrix(row_labels, column_labels)
        temp = xp.dot(M.conj().transpose(), M)
        temp = temp - xp.eye(*temp.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_unitary(self, row_labels, column_labels=None, absolute_threshold=1e-10):
        row_labels, column_labels = normalize_and_complement_argument_labels(self.labels, row_labels, column_labels)
        if soujou(self.dims_of_labels_front(row_labels)) != soujou(self.dims_of_labels_back(column_labels)):
            return False
        M = self.to_matrix(row_labels, column_labels)
        Mh = M.conj().transpose()
        temp = xp.dot(M, Mh)
        temp = temp - xp.eye(*temp.shape)
        if xp.linalg.norm(temp) > absolute_threshold:
            return False
        temp = xp.dot(Mh, M)
        temp = temp - xp.eye(*temp.shape)
        if xp.linalg.norm(temp) > absolute_threshold:
            return False
        return True





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

    def __copy__(self):
        return self.copy(shallow=True)

    def __deepcopy__(self):
        return self.copy(shallow=False)

    def __repr__(self):
        return f"DiagonalTensor(data={self.data}, labels={self.labels})"

    def __str__(self):
        if self.size > 100:
            dataStr = \
            "["*self.ndim + " ... " + "]"*self.ndim
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
    def dim(self):
        return self.data.shape[0]
    
    @property
    def shape(self): #tuple
        return (self.dim, self.dim)
    
    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    """
    #methods for moving indices
    #hobo muimi
    @outofplacable_tensorMixin_method
    def move_index_to_top(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)

    @outofplacable_tensorMixin_method
    def move_index_to_bottom(self, labelMove):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)

    @outofplacable_tensorMixin_method
    def move_index_to_position(self, labelMove, position, inplace=True):
        indexMoveFrom = self.index_of_label(labelMove)
        self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)

    @outofplacable_tensorMixin_method
    def move_indices_to_top(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)
        self.labels = labelsMove + diff_list(self.labels, labelsMove)

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, labelsMove):
        labelsMove = normalize_argument_labels(labelsMove)
        self.labels = diff_list(self.labels, labelsMove) + labelsMove

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normalize_argument_labels(labelsMove)
        labelsNotMove = diff_list(self.labels, labelsMove)
        self.labels = labelsNotMove[:position] + labelsMove + labelsNotMove[position:]

    @outofplacable_tensorMixin_method
    def move_all_indices(self, newLabels):
        newLabels = normalize_argument_labels(newLabels)
        oldLabels = self.labels
        if sorted(newLabels) != sorted(oldLabels):
            raise ValueError(f"newLabels do not match oldLabels. oldLabels=={oldLabels}, newLabels=={newLabels}")
        self.labels = newLabels
    """


    #methods for basic operations
    def __mul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(self, other)
        elif xp.isscalar(other):
            return DiagonalTensor(self.data*other, labels=self.labels)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(other, self)
        elif xp.isscalar(other):
            return DiagonalTensor(self.data*other, labels=self.labels)
        return NotImplemented

    def __truediv__(self, other):
        if xp.isscalar(other):
            return DiagonalTensor(self.data/other, labels=self.labels)
        return NotImplemented

    def __add__(self, other, skipLabelSort=False):
        if type(other)==DiagonalTensor:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            return DiagonalTensor(self.data+other.data, self.labels)
        return NotImplemented

    def __sub__(self, other, skipLabelSort=False):
        if type(other)==DiagonalTensor:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            return DiagonalTensor(self.data-other.data, self.labels)
        return NotImplemented

    def __eq__(self, other, skipLabelSort=False, absolute_threshold=1e-10):
        if type(other)==DiagonalTensor:
            diff = self.__sub__(other, skipLabelSort=skipLabelSort)
            return diff.norm() <= absolute_threshold
        return NotImplemented


    @inplacable_tensorMixin_method
    def conjugate(self):
        return DiagonalTensor(data=self.data.conj(),labels=self.labels)

    conj = conjugate

    def norm(self): #Frobenius norm
        return xp.linalg.norm(self.data)

    @inplacable_tensorMixin_method
    def normalize(self):
        norm = self.norm()
        return self / norm

    @inplacable_tensorMixin_method
    def inv(self):
        return DiagonalTensor(1.0/self.data, labels=self.labels)

    @inplacable_tensorMixin_method
    def sqrt(self):
        return DiagonalTensor(xp.sqrt(self.data), labels=self.labels)


    #methods for trace, contract
    def contract_internal(self, label1, label2):
        index1, index2 = tuple(indexs_duplable_front(self.labels, [label1, label2]))
        index1, index2 = min(index1,index2), max(index1,index2)
        if not(index1==0 and index2==1):
            warnings.warn(f"DiagonalTensor.contract_internal(label1, label2) must be index1,index2=0,1. but label1=={label1}, label2=={label2}, tensor=={self}. ignore.")
        return Tensor(xp.sum(self.data), [])

    def contract_common_internal(self):
        if self.labels[0]==self.labels[1]:
            return Tensor(xp.sum(self.data), [])
        else:
            return self.copy()

    def trace(self, label1=None, label2=None):
        if label1 is None:
            return self.contract_common_internal()
        else:
            return self.contract_internal(label1, label2)

    tr = trace

    def contract(self, *args, **kwargs):
        return contract(self, *args, **kwargs)

    def __getitem__(self, *labels):
        if len(labels)==1 and isinstance(labels[0],list): #if called like as A[["a"]]
            labels = labels[0]
        else:
            labels = list(labels)
        return ToContract(self, labels)


    #methods for converting to simple linalg object
    def to_tensor(self):
        return diagonalTensor_to_tensor(self)

    def to_matrix(self, row_labels, column_labels=None):
        return tensor_to_matrix(self.to_tensor(), row_labels, column_labels)

    def to_vector(self, labels):
        return tensor_to_vector(self.to_tensor(), labels)

    def to_scalar(self):
        return tensor_to_scalar(self.to_tensor())

    def to_diagonalMatrix(self):
        return diagonalTensor_to_diagonalMatrix(self)

    """
    Methods not in DiagonalTensor:
        fuse_indices
        split_index
        pad_indices
        add_dummy_index
        remove_all_dummy_indices
    """





#contract functions
class ToContract:
    #A["a"]*B["b"] == contract(A,B,["a"],["b"])
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

    def __mul__(self, other):
        return contract(self.tensor, other.tensor, self.labels, other.labels)

def contract(aTensor, bTensor, aLabelsContract, bLabelsContract):
    #Be careful to cLabels must be unique
    aLabelsContract = normalize_argument_labels(aLabelsContract)
    bLabelsContract = normalize_argument_labels(bLabelsContract)

    aIndicesContract = aTensor.indices_of_labels_back(aLabelsContract)
    bIndicesContract = bTensor.indices_of_labels_front(bLabelsContract)

    aDimsContract = aTensor.dims_of_indices(aIndicesContract)
    bDimsContract = bTensor.dims_of_indices(bIndicesContract)

    if aDimsContract != bDimsContract:
        raise ValueError(f"Dimss in contraction do not match. aLabelsContract=={aLabelsContract}, bLabelsContract=={bLabelsContract}, aDimsContract=={aDimsContract}, bDimsContract=={bDimsContract}")

    cLabels = diff_list(aTensor.labels, aLabelsContract) + diff_list(bTensor.labels, bLabelsContract)

    if type(aTensor)==DiagonalTensor and len(aDimsContract)==0:
        aTensor = aTensor.to_tensor()
    if type(bTensor)==DiagonalTensor and len(bDimsContract)==0:
        bTensor = bTensor.to_tensor()

    if type(aTensor)==Tensor and type(bTensor)==Tensor:
        cData = xp.tensordot(aTensor.data, bTensor.data, (aIndicesContract, bIndicesContract))

    elif type(aTensor)==Tensor and type(bTensor)==DiagonalTensor:
        if len(bDimsContract)==1:
            aTensor_ = aTensor.move_index_to_bottom(aLabelsContract[0], inplace=False)
            cData = xp.multiply(aTensor_.data, bTensor.data)
        else: #len(bDimsContract)==2:
            aTensor_ = aTensor.move_indices_to_bottom(aLabelsContract, inplace=False)
            cData = xp.multiply(aTensor_.data, bTensor.data)
            cData = xp.trace(cData, axis1=aTensor.ndim-2, axis2=aTensor.ndim-1)

    elif type(aTensor)==DiagonalTensor and type(bTensor)==Tensor:
        if len(aDimsContract)==1:
            bTensor_ = bTensor.move_index_to_bottom(bLabelsContract[0], inplace=False)
            cData = xp.multiply(aTensor.data, bTensor_.data)
            cData = xp.rollaxis(cData, bTensor.ndim-1, 0)
        else: #len(aDimsContract)==2:
            bTensor_ = bTensor.move_indices_to_bottom(bLabelsContract, inplace=False)
            cData = xp.multiply(aTensor.data, bTensor_.data)
            cData = xp.trace(cData, axis1=bTensor.ndim-2, axis2=bTensor.ndim-1)

    elif type(aTensor)==DiagonalTensor and type(bTensor)==DiagonalTensor:
        if len(aDimsContract)==1:
            cData = xp.multiply(aTensor.data, bTensor.data)
            return DiagonalTensor(cData, cLabels)
        else: #len(aDimsContract)==2:
            cData = xp.sum(xp.multiply(aTensor.data, bTensor.data))

    else:
        return NotImplemented

    return Tensor(cData, cLabels)


def contract_common(aTensor, bTensor):
    aLabels = aTensor.labels
    bLabels = bTensor.labels
    commonLabels = intersection_list(aLabels, bLabels)
    return contract(aTensor, bTensor, commonLabels, commonLabels)

def direct_product(aTensor, bTensor):
    return contract(aTensor, bTensor, [], [])





#converting functions
def tensor_to_matrix(tensor, row_labels, column_labels=None):
    row_labels, column_labels = normalize_and_complement_argument_labels(tensor.labels, row_labels, column_labels)

    t = tensor.move_all_indices(row_labels+column_labels, inplace=False)
    total_row_dim = soujou(t.shape[:len(row_labels)], dtype=int)
    total_column_dim = soujou(t.shape[len(row_labels):], dtype=int)

    return xp.reshape(t.data, (total_row_dim, total_column_dim))

def matrix_to_tensor(matrix, shape, labels):
    return Tensor(xp.reshape(matrix, shape), labels)

def tensor_to_vector(tensor, labels):
    t = tensor.move_all_indices(labels, inplace=False)
    return xp.reshape(t.data, (t.size,))

def vector_to_tensor(vector, shape, labels):
    return Tensor(xp.reshape(vector, shape), labels)

def tensor_to_scalar(tensor):
    return tensor.data.item(0)

def scalar_to_tensor(scalar):
    return Tensor(scalar, [])


def diagonalTensor_to_tensor(diagonalTensor):
    return Tensor(xp.diag(diagonalTensor.data), diagonalTensor.labels)

def tensor_to_diagonalTensor(tensor):
    return DiagonalTensor(xp.diag(tensor.data), tensor.labels)

def diagonalMatrix_to_diagonalTensor(diagonalMatrix, labels):
    return DiagonalTensor(diagonalMatrix, labels)

def diagonalTensor_to_diagonalMatrix(diagonalTensor):
    return diagonalTensor.data