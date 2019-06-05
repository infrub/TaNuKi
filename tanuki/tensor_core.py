from tanuki.tnxp import xp as xp
import copy as copyModule
import warnings
from numpy import prod as soujou
import textwrap
from collections import OrderedDict



#label :== string | tuple[label]
def normalize_argument_labels(labels):
    if isinstance(labels, list):
        return labels
    else:
        return [labels]

def normalize_and_complement_argument_labels(tensor, row_labels, column_labels=None):
    row_labels = normalize_argument_labels(row_labels)
    if column_labels is None:
        column_labels = [label for label in tensor.labels if label not in row_labels]
    else:
        column_labels = normalize_argument_labels(column_labels)
    return row_labels, column_labels

#decorate in-place Tensor class method with @outofplacable to be able to use as out-of-place method.
def outofplacable(f):
    def g(self, *args, inplace=True, **kwargs):
        if inplace:
            return f(self, *args, **kwargs)
        else:
            copied = self.copy(shallow=True)
            f(copied, *args, **kwargs)
            return copied
    return g

#decorate out-of-place Tensor class method with @inplacable to be able to use as out-of-place method.
def inplacable(f):
    def g(self, *args, inplace=False, **kwargs):
        if inplace:
            re = f(self, *args, **kwargs)
            self.data = re.data
            self.labels = re.labels
        else:
            return f(self, *args, **kwargs)
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
        if soujou(self.shape) > 100:
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

    def get_labels(self):
        return self._labels

    def set_labels(self, labels):
        if len(labels) != len(self.shape):
            raise ValueError(f"labels do not match shape of data. labels=={labels}, shape=={self.shape}")
        if len(labels) != len(set(labels)):
            raise ValueError(f"labels are not unique. labels=={labels}")
        self._labels = list(labels)

    labels = property(get_labels, set_labels)


    def index_of_label(self, label):
        return self.labels.index(label)

    def dim_of_index(self, index):
        return self.shape[index]

    def dim_of_label(self, label):
        return self.dim_of_index(self.index_of_label(label))

    def indices_of_labels(self,labels): #list[int]
        return [self.index_of_label(label) for label in labels]

    def dims_of_indices(self,indices): #tuple[int]
        return tuple(self.dim_of_index(index) for index in indices)

    def dims_of_labels(self,labels): #tuple[int]
        return self.dims_of_indices(self.indices_of_labels(labels))

    def replace_label(self, oldLabels, newLabels):
        oldLabels = normalize_argument_labels(oldLabels)
        newLabels = normalize_argument_labels(newLabels)
        for i, label in enumerate(self.labels):
            if label in oldLabels:
                self.labels[i] = newLabels[oldLabels.index(label)]


    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    #TODO pass if newIndices==oldIndices
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

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
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

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
        newIndicesMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom), self.ndim))

        oldIndicesNotMoveFrom = [i for i in range(len(self.labels)) if not i in oldIndicesMoveFrom]
        #newIndicesNotMoveTo = list(range(self.ndim-len(oldIndicesMoveFrom)))

        oldLabels = self.labels
        newLabels = [oldLabels[oldIndex] for oldIndex in oldIndicesNotMoveFrom] + [oldLabels[oldIndex] for oldIndex in oldIndicesMoveFrom]

        self.data = xp.moveaxis(self.data, oldIndicesMoveFrom, newIndicesMoveTo)
        self.labels = newLabels

    @outofplacable
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normalize_argument_labels(labelsMove)

        oldIndicesMoveFrom = self.indices_of_labels(labelsMove)
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

        if set(newLabels) != set(oldLabels):
            raise ValueError(f"newLabels do not match oldLabels. oldLabels=={oldLabels}, newLabels=={newLabels}")

        #oldPositions = list(range(self.ndim))
        newPositions = [newLabels.index(label) for label in oldLabels]

        self.data = xp.transpose(self.data, newPositions)
        self.labels = newLabels


    #methods for fuse/split
    #if new.. is no specified, assume like following:
    #["a","b","c","d"] <=split / fuse=> ["a",("b","c"),"d"]
    @outofplacable
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

    @outofplacable
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


    #methods for simple operation
    @inplacable
    def conjugate(self):
        return Tensor(data=self.data.conj(),labels=self.labels)

    conj = conjugate

    def __imul__(self, scalar):
        try:
            self.data *= scalar
            return self
        except:
            return NotImplemented

    def __mul__(self, scalar):
        try:
            out = Tensor(self.data*scalar, labels=self.labels)
            return out
        except:
            return NotImplemented

    def __rmul__(self, scalar):
        try:
            out = Tensor(self.data*scalar, labels=self.labels)
            return out
        except:
            return NotImplemented

    def __itruediv__(self, scalar):
        try:
            self.data /= scalar
            return self
        except:
            return NotImplemented

    def __truediv__(self, scalar):
        try:
            out = Tensor(self.data/scalar, labels=self.labels)
            return out
        except:
            return NotImplemented

    def __add__(self, other, skipLabelSort=False):
        try:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            return Tensor(self.data+other.data, self.labels)
        except:
            return NotImplemented

    def __iadd__(self, other, skipLabelSort=False):
        try:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            self.data += other.data
            return self
        except:
            return NotImplemented

    def __sub__(self, other, skipLabelSort=False):
        try:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            return Tensor(self.data-other.data, self.labels)
        except:
            return NotImplemented

    def __isub__(self, other, skipLabelSort=False):
        try:
            if not skipLabelSort:
                other = other.move_all_indices(self.labels, inplace=False)
            self.data -= other.data
            return self
        except:
            return NotImplemented

    @inplacable
    def pad_indices(self, labels, npads):
        indices = self.indices_of_labels(labels)
        wholeNpad = [(0,0)] * self.ndim
        for index,npad in zip(indices, npads):
            wholeNpad[index] = npad
        newData = xp.pad(self.data, wholeNpad, mode="constant", constant_values=0)
        return Tensor(newData, labels=self.labels)

    def norm(self): #Frobenius norm
        return xp.linalg.norm(self.data)

    @outofplacable
    def normalize(self):
        norm = self.norm()
        self.data /= norm


    #methods for trace, contract
    @inplacable
    def contract_internal(self, label1, label2):
        index1 = self.index_of_label(label1)
        index2 = self.index_of_label(label2)
        index1,index2 = min(index1,index2), max(index1,index2)

        newData = xp.trace(self.data, axis1=index1, axis2=index2)
        newLabels = self.labels[:index1]+self.labels[index1+1:index2]+self.labels[index2+1:]

        return Tensor(newData, newLabels)

    trace = contract_internal
    tr = contract_internal

    @inplacable
    def contract(self, *args, **kwargs):
        return contract(self, *args, **kwargs)

    def __getitem__(self, *labels):
        if len(labels)==1 and isinstance(labels[0],list): #if called like as A[["a"]]
            labels = labels[0]
        else:
            labels = list(labels)
        return ToContract(self, labels)


    #methods for dummy index
    @outofplacable
    def add_dummy_index(self, label):
        self.data = self.data[xp.newaxis, :]
        self.labels.insert(0, label)

    @outofplacable
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
    def to_matrix(self, row_labels, column_labels=None):
        return tensor_to_matrix(self, row_labels, column_labels)

    def to_vector(self, labels):
        return tensor_to_vector(self, labels)








class ToContract:
    """
    A["a"]*B["b"] == contract(A,B,["a"],["b"])
    """
    def __init__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels

    def __mul__(self, other):
        return contract(self.tensor, other.tensor, self.labels, other.labels)


def contract(aTensor, bTensor, aLabelsContract, bLabelsContract):
    """
    Be careful to cLabels must be unique
    """
    aLabelsContract = normalize_argument_labels(aLabelsContract)
    bLabelsContract = normalize_argument_labels(bLabelsContract)

    aIndicesContract = aTensor.indices_of_labels(aLabelsContract)
    bIndicesContract = bTensor.indices_of_labels(bLabelsContract)

    aDimsContract = aTensor.dims_of_indices(aIndicesContract)
    bDimsContract = bTensor.dims_of_indices(bIndicesContract)

    if aDimsContract != bDimsContract:
        raise ValueError(f"Dimss in contraction do not match. aLabelsContract=={aLabelsContract}, bLabelsContract=={bLabelsContract}, aDimsContract=={aDimsContract}, bDimsContract=={bDimsContract}")

    cData = xp.tensordot(aTensor.data, bTensor.data, (aIndicesContract, bIndicesContract))
    cLabels = [label for label in aTensor.labels if label not in aLabelsContract] + [label for label in bTensor.labels if label not in bLabelsContract]

    return Tensor(cData, cLabels)


def direct_product(aTensor, bTensor):
    return contract(aTensor, bTensor, [], [])




def tensor_to_matrix(tensor, row_labels, column_labels=None):
    row_labels, column_labels = normalize_and_complement_argument_labels(tensor, row_labels, column_labels)

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

"""
def tensor_to_scalar(tensor):
    if not tensor.size != 1:
        raise ValueError(f"tensor.size != 1 can't be treated as a scalar. size=={tensor.size}")

def scalar_to_tensor(scalar, labels):
"""


#I believe gesvd and gesdd return s which is positive, descending #TODO check
#A = (U["svd_ur"]*S["svd_sl"])["svd_sr"]*V["svd_vl"]
def tensor_svd(A, row_labels, column_labels=None, svd_label="svd_"):
    row_labels, column_labels = normalize_and_complement_argument_labels(tensor, row_labels, column_labels)

    row_dims = A.dims_of_labels(row_labels)
    column_dims = A.dims_of_labels(column_labels)

    a = A.to_matrix(row_labels, column_labels)

    try:
        u, s_diag, v = xp.linalg.svd(a, full_matrices=False)
    except (xp.linalg.LinAlgError, ValueError):
        warnings.warn("xp.linalg.svd failed with gesdd. retry with gesvd.")
        try:
            u, s_diag, v = xp.linalg.svd(a, full_matrices=False, lapack_driver="gesvd")
        except ValueError:
            raise 

    mid_dim = s_diag.shape[0]

    U = matrix_to_tensor(u, row_dims+(mid_dim,), row_labels+[svd_label+"ur"])
    S = matrix_to_tensor(xp.diag(s_diag), (mid_dim,mid_dim), [svd_label+"sl", svd_label+"sr"])
    V = matrix_to_tensor(v, (mid_dim,)+column_dims, [svd_label+"vl"]+column_labels)

    return U, S, V


def truncated_svd(A, row_labels, column_labels=None, chi=None, absolute_threshold=None, relative_threshold=None, svd_label="svd_"):
    U, S, V = tensor_svd(A, row_labels, column_labels, svd_label=svd_label)
    s_diag = xp.diag(S.data)

    if chi:
        trunc_s_diag = s_diag[:chi]

    if absolute_threshold:
        trunc_s_diag = trunc_s_diag[trunc_s_diag > absolute_threshold]

    if relative_threshold:
        threshold = relative_threshold * s_diag[0]
        trunc_s_diag = trunc_s_diag[trunc_s_diag > threshold]

    chi = len(trunc_s_diag)

    S.data = xp.diag(trunc_s_diag)
    U.move_indices_to_top(svd_label+"ur")
    U.data = U.data[0:chi]
    U.move_indices_to_bottom(svd_label+"ur")
    V.data = V.data[0:chi]

    return U, S, V


#A = Q["qr_qr"]*R["qr_rl"]
def tensor_qr(A, row_labels, column_labels=None, qr_label="qr_", mode="economic"):
    row_labels, column_labels = normalize_and_complement_argument_labels(A, row_labels, column_labels)

    row_dims = A.dims_of_labels(row_labels)
    column_dims = A.dims_of_labels(column_labels)

    a = A.to_matrix(row_labels, column_labels)

    q, r = xp.linalg.qr(a, mode=mode)

    mid_dim = r.shape[0]

    Q = matrix_to_tensor(q, row_dims+(mid_dim,), row_labels+[qr_label+"qr"])
    R = matrix_to_tensor(r, (mid_dim,)+column_dims, [qr_label+"rl"]+column_labels)

    return Q, R


#A = L["lq_lr"]*Q["lq_ql"]
def tensor_lq(A, row_labels, column_labels=None, lq_label="lq_", mode="economic"):
    row_labels, column_labels = normalize_and_complement_argument_labels(A, row_labels, column_labels)

    temp_label = "qr_in_lq_temp_label"

    Q, L = tensor_qr(A, column_labels, row_labels, qr_label=temp_label, mode=mode)

    Q.replace_label(temp_label+"qr", lq_label+"ql")
    L.replace_label(temp_label+"rl", lq_label+"lr")

    return L, Q

#A = Vh["eigh_vhr"]*W["eigh_wl"]["eigh_wr"]*V[eigh_vl]