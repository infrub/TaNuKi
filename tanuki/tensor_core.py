from tanuki.tnxp import xp as xp
from tanuki.utils import *
from tanuki.errors import *
from tanuki.decomp import *
import copy as copyModule
import warnings
import textwrap
from collections import OrderedDict
import uuid
import random
from math import sqrt

display_max_size = 100



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









#converting functions
def tensor_to_ndarray(T, indices):
    T = T.move_all_indices(indices)
    return T.data

def tensor_to_matrix(T, rows=None, cols=None):
    rows, cols = T.normarg_complement_indices(rows, cols)
    T = T.move_all_indices(rows+cols)
    total_row_dim = soujou(T.shape[:len(rows)])
    total_col_dim = soujou(T.shape[len(rows):])
    return xp.reshape(T.data, (total_row_dim, total_col_dim))

def tensor_to_vector(T, indices):
    T = T.move_all_indices(indices)
    return xp.reshape(T.data, (T.size,))

def tensor_to_scalar(T):
    assert T.size == 1
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
    return Tensor(xp.diagflat(DT.data).reshape(shape), DT.labels)

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





#confirming functions
def matrix_is_diagonal(M, check_rtol=1e-5, check_atol=1e-8):
    if M.ndim != 2:
        return CollateralBool(False, {"reason":"NOT_MATRIX"})
    zeros = xp.zeros(M.shape)
    eye = xp.eye(*M.shape)
    notDiagonals = xp.where(eye==zeros, M, zeros)
    if not xp.allclose(notDiagonals, zeros, rtol=check_rtol, atol=check_atol):
        return CollateralBool(False, {"reason":"NOT_DIAGONAL"})
    return CollateralBool(True, {})

def matrix_is_identity(M, check_rtol=1e-5, check_atol=1e-8):
    if M.ndim != 2:
        return CollateralBool(False, {"reason":"NOT_MATRIX"})
    eye = xp.eye(*M.shape)
    if not xp.allclose(M, eye, rtol=check_rtol, atol=check_atol):
        return CollateralBool(False, {"reason":"NOT_IDENTITY"})
    return CollateralBool(True, {})

def matrix_is_prop_identity(M, check_rtol=1e-5, check_atol=1e-8):
    hoge = matrix_is_diagonal(M, check_rtol=check_rtol, check_atol=check_atol)
    if not hoge: return hoge
    d = xp.diagonal(M)
    factor = xp.average(d)
    ones = xp.ones_like(d)
    if not xp.allclose(d, factor*ones, rtol=check_rtol, atol=check_atol):
        return CollateralBool(False, {"reason":"NOT_PROP_IDENTITY", "factor":factor})
    return CollateralBool(True, {"factor":factor})

def matrix_is_left_semi_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    N = xp.dot(M.conj().transpose(), M)
    return matrix_is_identity(N, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_right_semi_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    N = xp.dot(M, M.conj().transpose())
    return matrix_is_identity(N, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    return matrix_is_left_semi_unitary(M, check_rtol=check_rtol, check_atol=check_atol) & matrix_is_right_semi_unitary(M, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_prop_left_semi_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    N = xp.dot(M.conj().transpose(), M)
    return matrix_is_prop_identity(N, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_prop_right_semi_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    N = xp.dot(M, M.conj().transpose())
    return matrix_is_prop_identity(N, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_prop_unitary(M, check_rtol=1e-5, check_atol=1e-8):
    return matrix_is_prop_left_semi_unitary(M, check_rtol=check_rtol, check_atol=check_atol) & matrix_is_prop_right_semi_unitary(M, check_rtol=check_rtol, check_atol=check_atol)

def matrix_is_triu(M, check_rtol=1e-5, check_atol=1e-8):
    return CollateralBool(xp.allclose(M, xp.triu(M), rtol=check_rtol, atol=check_atol), {})

def matrix_is_tril(M, check_rtol=1e-5, check_atol=1e-8):
    return CollateralBool(xp.allclose(M, xp.tril(M), rtol=check_rtol, atol=check_atol), {})

def ndarray_is_prop_to(M, I, check_rtol=1e-5, check_atol=1e-8):
    if M.shape != I.shape:
        return CollateralBool(False, {"reason":"WRONG_SHAPE"})
    Ih = I.conj()
    c = xp.sum(M*Ih) / xp.sum(I*Ih)

    if not xp.allclose(M, I*c, rtol=check_rtol, atol=check_atol):
        return CollateralBool(False, {"reason":"NOT_PROP_TO", "factor":c})

    return CollateralBool(True, {"factor":c})





#classes
class TensorLabelingMixin:
    def get_labels(self):
        return self._labels
    def set_labels(self, labels):
        if len(labels) != self.ndim:
            raise LabelsLengthError(f"{labels}, {self.shape}")
        if not is_type_labels(labels):
            raise LabelsTypeError(f"{labels}")
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



    #methods for normarg
    def normarg_index_front(self, indexOrLabel):
        if type(indexOrLabel)==int:
            return indexOrLabel
        elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
            return self.index_of_label_front(indexOrLabel)
        else:
            raise LabelsTypeError(f"indexOrLabel={indexOrLabel}")

    def normarg_index_back(self, indexOrLabel):
        if type(indexOrLabel)==int:
            return indexOrLabel
        elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
            return self.index_of_label_back(indexOrLabel)
        else:
            raise LabelsTypeError(f"indexOrLabel={indexOrLabel}")

    normarg_index = normarg_index_front

    def normarg_indices_front(self, indicesOrLabels):
        if isinstance(indicesOrLabels, TensorLabelingMixin):
            indicesOrLabels = intersection_list(self.labels, indicesOrLabels.labels)
        if type(indicesOrLabels)!=list:
            indicesOrLabels = [indicesOrLabels]
        res = [None] * len(indicesOrLabels)
        temp = list(self.labels)
        for rei, indexOrLabel in enumerate(indicesOrLabels):
            if type(indexOrLabel)==int:
                index = indexOrLabel
            else:
                continue
            res[rei] = index
            temp[index] = None
        for rei, indexOrLabel in enumerate(indicesOrLabels):
            if type(indexOrLabel)==int:
                continue
            elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
                index = temp.index(indexOrLabel)
            else:
                raise LabelsTypeError(f"indexOrLabel={indexOrLabel}")
            res[rei] = index
            temp[index] = None
        return res

    def normarg_indices_back(self, indicesOrLabels):
        if isinstance(indicesOrLabels, TensorLabelingMixin):
            indicesOrLabels = intersection_list(self.labels, indicesOrLabels.labels)
        if type(indicesOrLabels)!=list:
            indicesOrLabels = [indicesOrLabels]
        res = [None] * len(indicesOrLabels)
        indicesOrLabels = list(reversed(indicesOrLabels))
        temp = list(reversed(self.labels))
        for rei,indexOrLabel in enumerate(indicesOrLabels):
            if type(indexOrLabel)==int:
                index = indexOrLabel
                revindex = len(temp)-1-index
            else:
                continue
            res[rei] = index
            temp[revindex] = None
        for rei,indexOrLabel in enumerate(indicesOrLabels):
            if type(indexOrLabel)==int:
                continue
            elif type(indexOrLabel)==tuple or type(indexOrLabel)==str:
                revindex = temp.index(indexOrLabel)
                index = len(temp)-1-revindex
            else:
                raise LabelsTypeError(f"indexOrLabel={indexOrLabel}")
            res[rei] = index
            temp[revindex] = None
        return list(reversed(res))

    normarg_indices = normarg_indices_front

    def normarg_complement_indices(self, rows=None, cols=None):
        if rows is None and cols is None:
            raise ValueError
        elif cols is None:
            rows = self.normarg_indices_front(rows)
            cols = diff_list(list(range(self.ndim)), rows)
        elif rows is None:
            cols = self.normarg_indices_back(cols)
            rows = diff_list(list(range(self.ndim)), cols)
        else:
            rows = self.normarg_indices_front(rows)
            cols = self.normarg_indices_back(cols)
        return rows, cols




    #methods for changing labels
    @outofplacable_tensorMixin_method
    def replace_labels(self, oldIndices, newLabels):
        oldIndices = self.normarg_indices(oldIndices)
        newLabels = normarg_labels(newLabels)
        for oldIndex, newLabel in zip(oldIndices, newLabels):
            self.labels[oldIndex] = newLabel

    @outofplacable_tensorMixin_method
    def aster_labels(self, oldIndices=None):
        if oldIndices is None: oldIndices=self.labels
        oldIndices = self.normarg_indices(oldIndices)
        oldLabels = self.labels_of_indices(oldIndices)
        newLabels = aster_labels(oldLabels)
        self.replace_labels(oldIndices, newLabels)

    @outofplacable_tensorMixin_method
    def unaster_labels(self, oldIndices=None):
        if oldIndices is None: oldIndices=self.labels
        oldIndices = self.normarg_indices(oldIndices)
        oldLabels = self.labels_of_indices(oldIndices)
        newLabels = unaster_labels(oldLabels)
        self.replace_labels(oldIndices, newLabels)

    @outofplacable_tensorMixin_method
    def assign_labels(self, base_label):
        self.labels = [base_label+"_"+str(i) for i in range(self.ndim)]

    def enunique_labels(self, oldIndices):
        newLabels = [unique_label() for _ in oldIndices]
        self.replace_labels(oldIndices, newLabels)
        return newLabels




class TensorMixin(TensorLabelingMixin):
    def __copy__(self):
        return self.copy(shallow=True)

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = {}
        return self.copy(shallow=False)



    #methods for basic unary operations
    def norm(self): #Frobenius norm
        return xp.linalg.norm(self.data)

    def normalize(self, inplace=False):
        norm = self.norm()
        if inplace:
            self.data = self.data / norm
            return norm
        else:
            return self / norm

    def __abs__(self):
        return self.norm()

    def __neg__(self):
        return self.__class__(-self.data, labels=self.labels)

    @inplacable_tensorMixin_method
    def real(self):
        return self.__class__(xp.real(self.data), self.labels)

    @inplacable_tensorMixin_method
    def imag(self):
        return self.__class__(xp.imag(self.data), self.labels)

    @inplacable_tensorMixin_method
    def abs(self):
        return self.__class__(xp.absolute(self.data), self.labels)

    @inplacable_tensorMixin_method
    def conjugate(self):
        return self.__class__(self.data.conj(), labels=self.labels)

    conj = conjugate

    @inplacable_tensorMixin_method
    def transpose(self, rows=None, cols=None):
        rows, cols = self.normarg_complement_indices(rows, cols)
        row_labels, col_labels = self.labels_of_indices(rows), self.labels_of_indices(cols)
        if len(rows) != len(cols):
            raise IndicesLengthError(f"rows=={rows}, cols=={cols}")
        out = self.copy()
        out.replace_labels(rows+cols, col_labels+row_labels)
        return out
        
    @inplacable_tensorMixin_method
    def adjoint(self, rows=None, cols=None, style="aster"):
        if style=="transpose":
            rows, cols = self.normarg_complement_indices(rows, cols)
            row_labels, col_labels = self.labels_of_indices(rows), self.labels_of_indices(cols)
            if len(rows) != len(cols):
                raise IndicesLengthError(f"rows=={rows}, cols=={cols}")
            out = self.conjugate()
            out.replace_labels(rows+cols, col_labels+row_labels)
        elif style=="aster":
            if rows is None:
                rows = [i for i in range(self.ndim)]
            if cols is None:
                cols = []
            rows = self.normarg_indices_front(rows)
            cols = self.normarg_indices_back(cols)
            out = self.conjugate()
            out.aster_labels(rows+cols)
        return out

    adj = adjoint

    @inplacable_tensorMixin_method
    def hermite(self, rows, cols=None, assume_definite_and_if_negative_then_make_positive=False):
        re = (self + self.adjoint(rows,cols,style="transpose"))/2
        if assume_definite_and_if_negative_then_make_positive:
            if xp.real(re.data.item(0)) < 0:
                re = re * (-1)
        return re

    @inplacable_tensorMixin_method
    def antihermite(self, rows, cols=None):
        return (self - self.adjoint(rows,cols,style="transpose"))/2



    #methods for basic binary operations
    def __add__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if isinstance(self, DiagonalTensor) and isinstance(other, DiagonalTensor):
                try:
                    if not skipLabelSort:
                        other = other.move_all_indices(self.labels)
                    return DiagonalTensor(self.data+other.data, self.labels)
                except:
                    pass
            if isinstance(self, DiagonalTensor):
                self = self.to_tensor()
            if isinstance(other, DiagonalTensor):
                other = other.to_tensor()
            if not skipLabelSort:
                other = other.move_all_indices(self.labels)
            return Tensor(self.data+other.data, self.labels)
        return NotImplemented

    def __sub__(self, other, skipLabelSort=False):
        if isinstance(other, TensorMixin):
            if isinstance(self, DiagonalTensor) and isinstance(other, DiagonalTensor):
                try:
                    if not skipLabelSort:
                        other = other.move_all_indices(self.labels)
                    return DiagonalTensor(self.data-other.data, self.labels)
                except:
                    pass
            if isinstance(self, DiagonalTensor):
                self = self.to_tensor()
            if isinstance(other, DiagonalTensor):
                other = other.to_tensor()
            if not skipLabelSort:
                other = other.move_all_indices(self.labels)
            return Tensor(self.data-other.data, self.labels)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(self, other)
        elif xp.isscalar(other):
            return self.__class__(self.data*other, labels=self.labels)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, TensorMixin):
            return contract_common(other, self)
        elif xp.isscalar(other):
            return self.__class__(self.data*other, labels=self.labels)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return other.solve(self)
        elif isinstance(other, DiagonalTensor):
            return self * other.inv()
        elif xp.isscalar(other):
            return self.__class__(self.data/other, labels=self.labels)
        return NotImplemented

    def __eq__(self, other, skipLabelSort=False, check_rtol=1e-5, check_atol=1e-8):
        if isinstance(other, TensorMixin):
            if isinstance(self, DiagonalTensor) and isinstance(other, DiagonalTensor):
                try:
                    if not skipLabelSort:
                        other = other.move_all_indices(self.labels)
                    return xp.allclose(self.data, other.data, rtol=check_rtol, atol=check_atol)
                except:
                    pass
            if isinstance(self, DiagonalTensor):
                self = self.to_tensor()
            if isinstance(other, DiagonalTensor):
                other = other.to_tensor()
            if not skipLabelSort:
                try:
                    other = other.move_all_indices(self.labels)
                except:
                    return False
            if self.shape != other.shape:
                return False
            return xp.allclose(self.data, other.data, rtol=check_rtol, atol=check_atol)
        elif xp.isscalar(other):
            if self.ndim == 0:
                return xp.allclose(self.to_scalar(), other, rtol=check_rtol, atol=check_atol)
            else:
                return False
        return NotImplemented



    #methods for contract
    def __getitem__(self, kusonmiteenashorisaretahikisuu):
        # kusonmiteenashorisaretahikisuu is really shit.
        #
        # input   => kusonmiteenashorisaretahikisuu => indices
        # []      => error          => []
        # [x]     => x              => [x]
        # [x,y]   => (x,y) # fuck   => [x,y]
        # [()]    => ()             => [()]
        # [(x,)]  => (x,)           => [(x,)]
        # [(x,y)] => (x,y) # fuck   => [(x,y)]
        # [[]]    => []             => []
        # [[x]]   => [x]            => [x]
        # [[x,y]] => [x,y]          => [x,y]
        #
        # so inevitable ambiguousity exist in some cases e.g. calling [(x,y)] vs [x,y] when labels==[x,y,(x,y)]
        if isinstance(kusonmiteenashorisaretahikisuu, slice):
            return ToContract(self, kusonmiteenashorisaretahikisuu.start, kusonmiteenashorisaretahikisuu.stop, kusonmiteenashorisaretahikisuu.step)
        if isinstance(kusonmiteenashorisaretahikisuu, list):
            indices = kusonmiteenashorisaretahikisuu
        elif type(kusonmiteenashorisaretahikisuu)==tuple:
            if kusonmiteenashorisaretahikisuu in self.labels:
                indices = [kusonmiteenashorisaretahikisuu]
            else:
                indices = list(kusonmiteenashorisaretahikisuu)
        else:
            indices = [kusonmiteenashorisaretahikisuu]

        return ToContract(self, indices)



    #confirming methods
    def is_diagonal(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_diagonal(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_identity(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_identity(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_identity(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_prop_identity(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_left_semi_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_left_semi_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_right_semi_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_right_semi_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_semi_unitary(T, inds, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_left_semi_unitary(T.to_matrix(inds, None), check_rtol=check_rtol, check_atol=check_atol)

    def is_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_left_semi_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_prop_left_semi_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_right_semi_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_prop_right_semi_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_semi_unitary(T, inds, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_prop_left_semi_unitary(T.to_matrix(inds, None), check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_unitary(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        return matrix_is_prop_unitary(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_triu(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8): # row_index <= col_index
        return matrix_is_triu(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_tril(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8): # row_index >= col_index
        return matrix_is_tril(T.to_matrix(rows, cols), check_rtol=check_rtol, check_atol=check_atol)

    def is_hermite(T, rows, cols=None, check_rtol=1e-5, check_atol=1e-8):
        Th = T.adjoint(rows, cols, style="transpose")
        return T.__eq__(Th, check_rtol=check_rtol, check_atol=check_atol)

    def is_prop_to(self, other, skipLabelSort=False, check_rtol=1e-5, check_atol=1e-8):
        if not skipLabelSort:
            other = other.move_all_indices(self.labels)
        return ndarray_is_prop_to(self.data, other.data, check_rtol=check_rtol, check_atol=check_atol)

    def is_dummy(self):
        return self.data.ndim==0 and self.data.item(0)==1.0





class Tensor(TensorMixin):
    #basic methods
    def __init__(self, data, labels=None, base_label=None, copy=False):
        self.data = xp.array(data, copy=copy)
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

    def __str__(self, nodata=False):
        if nodata:
            return \
            f"Tensor(\n" + \
            f"    labels={self.labels},\n" + \
            f"    shape={self.shape},\n" + \
            f")"

        if self.size > display_max_size:
            dataStr = \
            "["*self.ndim + str(self.data.item(0)) + " ... " + "]"*self.ndim
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



    #methods for basic binary operations
    def round(self, decimals=8):
        return Tensor(self.data.round(decimals=decimals), self.labels)



    #methods for truncate, pad, dummy
    @inplacable_tensorMixin_method
    def truncate_index(self, index, fromto1, fromto2=None):
        index = self.normarg_index(index)
        if fromto2 is None:
            fromto2 = fromto1
            fromto1 = 0
        data = self.data
        data = xp.split(data, [fromto1, fromto2], axis=index)[1]
        return Tensor(data, self.labels)

    @inplacable_tensorMixin_method
    def pad_indices(self, indices, npads):
        indices = self.normarg_indices(indices)
        wholeNpad = [(0,0)] * self.ndim
        for index, npad in zip(indices, npads):
            wholeNpad[index] = npad
        newData = xp.pad(self.data, wholeNpad, mode="constant", constant_values=0)
        return Tensor(newData, labels=self.labels)

    @inplacable_tensorMixin_method
    def add_dummy_index(self, label=()):
        newData = self.data[xp.newaxis, :]
        newLabels = [label] + self.labels
        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def remove_dummy_indices(self, indices=None):
        if indices is not None: indices = self.normarg_indices(indices)
        oldShape = self.shape
        oldLabels = self.labels
        newShape = ()
        newLabels = []
        for i, x in enumerate(oldLabels):
            if oldShape[i]==1 and (indices is None or i in indices):
                pass
            else:
                newShape = newShape + (oldShape[i],)
                newLabels.append(x)
        newData = self.data.reshape(newShape)
        return Tensor(newData, newLabels)



    #methods for moving indices
    #I assumed that rollaxis is better than moveaxis in terms of computing costs
    #TODO pass if newIndices==oldIndices
    """
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
    """
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
            raise IndicesLengthError()
        moveTo = list(range(self.ndim))
        newLabels = self.labels_of_indices(moveFrom)
        newData = xp.moveaxis(self.data, moveFrom, moveTo)
        return Tensor(newData, newLabels)


    
    #methods for fuse/split
    #if new.. is no specified, assume like following:
    #["a","b","c","d"] <=split / fuse=> ["a",("b","c"),"d"]
    @inplacable_tensorMixin_method
    def fuse_indices(self, splittedLabels=None, fusedLabel=None, input_memo=None, output_memo=None):
        if input_memo is None:
            input_memo = {}
        if output_memo is None:
            output_memo = {}

        if splittedLabels is None:
            if "splittedLabels" in input_memo:
                splittedLabels = input_memo["splittedLabels"]
            else:
                raise ValueError
        splittedIndices = self.normarg_indices(splittedLabels)
        splittedLabels = self.labels_of_indices(splittedIndices)

        if fusedLabel is None:
            if "fusedLabel" in input_memo:
                fusedLabel = input_memo["fusedLabel"]
            else:
                fusedLabel = tuple(splittedLabels)

        position = min(splittedIndices)
        self = self.move_indices_to_position(splittedIndices, position)
        del splittedIndices

        oldShape = self.shape
        splittedShape = oldShape[position:position+len(splittedLabels)]
        fusedDim = soujou(splittedShape)
        newShape = oldShape[:position] + (fusedDim,) + oldShape[position+len(splittedLabels):]

        oldLabels = self.labels
        newLabels = oldLabels[:position] + [fusedLabel] + oldLabels[position+len(splittedLabels):]
        newData = xp.reshape(self.data, newShape)

        output_memo.update({"splittedShape":splittedShape, "splittedLabels":splittedLabels, "fusedDim":fusedDim, "fusedLabel":fusedLabel})

        return Tensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def split_index(self, fusedLabel=None, splittedShape=None, splittedLabels=None, input_memo=None, output_memo=None):
        if input_memo is None:
            input_memo = {}
        if output_memo is None:
            output_memo = {}

        if fusedLabel is None:
            if "fusedLabel" in input_memo:
                fusedLabel = input_memo["fusedLabel"]
            else:
                raise ValueError
        fusedIndex = self.normarg_index(fusedLabel)
        fusedLabel = self.label_of_index(fusedIndex)

        if splittedShape is None:
            if "splittedShape" in input_memo:
                splittedShape = input_memo["splittedShape"]
            else:
                raise ValueError
        splittedShape = tuple(splittedShape)

        if splittedLabels is None:
            if "splittedLabels" in input_memo:
                splittedLabels = input_memo["splittedLabels"]
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
        newData = xp.reshape(self.data, newShape)

        output_memo.update({"fusedDim":fusedDim, "fusedLabel":fusedLabel, "splittedShape":splittedShape, "splittedLabels":splittedLabels})

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
    def contract_internal_indices(self, indices1, indices2):
        indices1 = self.normarg_indices_front(indices1)
        indices2 = self.normarg_indices_back(indices2)

        temp = self.copy(shallow=True)
        while len(indices1)!=0:
            index1 = indices1.pop()
            index2 = indices2.pop()
            index1, index2 = min(index1,index2), max(index1,index2)

            def dokoitta(x):
                if 0<=x<index1: return x
                elif index1<x<index2: return x-1
                elif index2<x<temp.ndim: return x-2
                else: raise IndexError()

            indices1 = [dokoitta(x) for x in indices1]
            indices2 = [dokoitta(x) for x in indices2]
            temp = temp.contract_internal_index(index1, index2)

        return temp

    @inplacable_tensorMixin_method
    def contract_internal_common(self):
        temp = self
        commons = floor_half_list(temp.labels)
        temp = temp.contract_internal_indices(commons, commons)
        return temp

    
    @inplacable_tensorMixin_method
    def contract_internal(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_internal_common()
        elif type(index1) == list:
            return self.contract_internal_indices(index1, index2)
        else:
            return self.contract_internal_index(index1, index2)

    trace = contract_internal
    tr = contract_internal



    #converting methods
    to_diagonalTensor = tensor_to_diagonalTensor
    to_ndarray = tensor_to_ndarray
    to_matrix = tensor_to_matrix
    to_vector = tensor_to_vector
    to_scalar = tensor_to_scalar

    def qr(self, *args, **kwargs):
        return tensor_qr(self, *args, **kwargs)

    def lq(self, *args, **kwargs):
        return tensor_lq(self, *args, **kwargs)

    def svd(self, *args, **kwargs):
        return tensor_svd(self, *args, **kwargs)

    def eigh(self, *args, **kwargs):
        return tensor_eigh(self, *args, **kwargs)

    def eigsh(self, *args, **kwargs):
        return tensor_eigsh(self, *args, **kwargs)

    def solve(self, *args, **kwargs):
        return tensor_solve(self, *args, **kwargs)


# A[i,j,k,l] = [i==k][j==l]A.data[i,j]
class DiagonalTensor(TensorMixin):
    #basic methods
    def __init__(self, data, labels=None, base_label=None, copy=False):
        self.data = xp.array(data, copy=copy)
        if labels is None:
            if base_label is None:
                base_label = unique_label()
            self.assign_labels(base_label)
        else:
            if isinstance(labels, list) and len(labels)==self.halfndim:
                labels = labels + labels
            self.labels = labels

    def copy(self, shallow=False):
        return DiagonalTensor(self.data, self.labels, copy=not(shallow))

    def __repr__(self):
        return f"DiagonalTensor(data={self.data}, labels={self.labels})"

    def __str__(self, nodata=False):
        if nodata:
            return \
            f"DiagonalTensor(\n" + \
            f"    labels={self.labels},\n" + \
            f"    shape={self.shape},\n" + \
            f")"

        if self.halfsize > display_max_size:
            dataStr = \
            "["*self.halfndim + str(self.data.item(0)) + " ... " + "]"*self.halfndim
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
    def shape(self):
        return self.halfshape + self.halfshape
    
    @property
    def ndim(self):
        return self.halfndim * 2

    @property
    def size(self):
        return self.halfsize * self.halfsize

    @property
    def dtype(self):
        return self.data.dtype



    #methods for basic binary operations
    def round(self, decimals=8):
        return Tensor(self.data.round(decimals=decimals), self.labels)

    @inplacable_tensorMixin_method
    def inv(self):
        return DiagonalTensor(1.0/self.data, labels=self.labels)

    @inplacable_tensorMixin_method
    def sqrt(self):
        return DiagonalTensor(xp.sqrt(self.data), labels=self.labels)

    @inplacable_tensorMixin_method
    def sqrt2(self, firstHalfIndices=None):
        if firstHalfIndices is None:
            temp = self
        else:
            firstHalfIndices = self.normarg_indices_front(firstHalfIndices)
            temp = self.move_half_all_indices_to_top(firstHalfIndices)
        firstHalfLabels = temp.labels[:temp.ndim//2]
        lastHalfLabels = temp.labels[temp.ndim//2:]
        newLabels = [unique_label() for _ in firstHalfLabels]
        nakami = xp.sqrt(temp.data)
        return DiagonalTensor(nakami, labels=firstHalfLabels+newLabels), DiagonalTensor(nakami, labels=newLabels+lastHalfLabels), 

    @inplacable_tensorMixin_method
    def exp(self, coeff=1):
        return DiagonalTensor(xp.exp(coeff*self.data), labels=self.labels)



    #methods for truncate, pad, dummy
    @inplacable_tensorMixin_method
    def truncate_index(self, index, fromto1, fromto2=None):
        index = self.normarg_index(index)
        if index >= self.halfndim:
            index = index - self.halfndim
        if fromto2 is None:
            fromto2 = fromto1
            fromto1 = 0
        data = self.data
        data = xp.split(data, [fromto1, fromto2], axis=index)[1]
        return DiagonalTensor(data, self.labels)



    # MUST bag{moveFrom}==bag{0:self.ndim} (else: idk)
    # WILL moveFrom keep diagonality (else: CantKeepDiagonalityError)
    @inplacable_tensorMixin_method
    def move_all_indices_assuming_can_keep_diagonality(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        if len(moveFrom)!=self.ndim:
            raise IndicesLengthError()

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
            raise IndicesLengthError()
        moveFrom = halfMoveFrom + [(x+self.halfndim)%self.ndim for x in halfMoveFrom]
        if not eq_list(moveFrom, list(range(self.ndim))):
            raise CantKeepDiagonalityError()
        return self.move_all_indices_assuming_can_keep_diagonality(moveFrom)



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
            return DiagonalTensor(newData, newLabels)

        coindex1, coindex2 = (index1+self.halfndim)%self.ndim, (index2+self.halfndim)%self.ndim
        halfindex1, halfindex2 = index1%self.halfndim, index2%self.halfndim
        halfindex1, halfindex2 = min(halfindex1, halfindex2), max(halfindex1, halfindex2)
        newData = xp.diagonal(self.data, axis1=halfindex1, axis2=halfindex2)

        newLabels = self.labels[0:halfindex1]+self.labels[halfindex1+1:halfindex2]+self.labels[halfindex2+1:self.halfndim] \
            + self.labels[coindex1:coindex1+1] \
            + self.labels[self.halfndim:self.halfndim+halfindex1]+self.labels[self.halfndim+halfindex1+1:self.halfndim+halfindex2]+self.labels[self.halfndim+halfindex2+1:self.ndim] \
            + self.labels[coindex2:coindex2+1]

        return DiagonalTensor(newData, newLabels)

    @inplacable_tensorMixin_method
    def contract_internal_indices(self, indices1, indices2):
        indices1 = self.normarg_indices_front(indices1)
        indices2 = self.normarg_indices_back(indices2)

        temp = self.copy(shallow=True)
        while len(indices1)!=0:
            index1 = indices1.pop()
            index2 = indices2.pop()
            index1, index2 = min(index1,index2), max(index1,index2)

            if index1+self.halfndim == index2:
                def dokoitta(x):
                    if 0<=x<index1: return x
                    elif index1<x<index2: return x-1
                    elif index2<x<self.ndim: return x-2
                    else: raise IndexError()
            else:
                coindex1, coindex2 = (index1+temp.halfndim)%temp.ndim, (index2+temp.halfndim)%temp.ndim
                halfindex1, halfindex2 = index1%temp.halfndim, index2%temp.halfndim
                halfindex1, halfindex2 = min(halfindex1, halfindex2), max(halfindex1, halfindex2)
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
            temp = temp.contract_internal_index(index1, index2)
        return temp

    @inplacable_tensorMixin_method
    def contract_internal_common(self):
        temp = self
        commons = floor_half_list(temp.labels)
        temp = temp.contract_internal_indices(commons, commons)
        return temp

    @inplacable_tensorMixin_method
    def contract_internal(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_internal_common()
        elif type(index1) == list:
            return self.contract_internal_indices(index1, index2)
        else:
            return self.contract_internal_index(index1, index2)

    trace = contract_internal
    tr = contract_internal



    #converting methods
    to_tensor = diagonalTensor_to_tensor
    to_diagonalElementsNdarray = diagonalTensor_to_diagonalElementsNdarray
    to_diagonalElementsVector = diagonalTensor_to_diagonalElementsVector
    to_matrix = diagonalTensor_to_matrix
    to_vector = diagonalTensor_to_vector
    to_scalar = diagonalTensor_to_scalar

    def __getattr__(self, name):
        return getattr(self.to_tensor(), name)





#contract functions
class ToContract:
    #A["a"]*B["b"] == contract(A,B,["a"],["b"])
    def __init__(self, tensor, labelsToContract, labelsRenameFrom=None, labelsRenameTo=None):
        self.tensor = tensor
        self.labelsToContract = labelsToContract
        self.labelsRenameFrom = labelsRenameFrom
        self.labelsRenameTo = labelsRenameTo

    def __mul__(self, other):
        result = contract(self.tensor, other.tensor, self.labelsToContract, other.labelsToContract)
        if self.labelsRenameFrom != None:
            result.replace_labels(self.labelsRenameFrom, self.labelsRenameTo, inplace=True)
        if other.labelsRenameFrom != None:
            result.replace_labels(other.labelsRenameFrom, other.labelsRenameTo, inplace=True)
        return result



def direct_product(A, B):
    if type(A)==Tensor and type(B)==Tensor:
        cData = xp.tensordot(A.data, B.data, 0)
        cLabels = A.labels + B.labels
        return Tensor(cData, cLabels)
    elif type(A)==DiagonalTensor and type(B)==DiagonalTensor:
        cData = xp.tensordot(A.data, B.data, 0)
        cLabels = A.labels[:A.halfndim] + B.labels[:B.halfndim] + A.labels[A.halfndim:] + B.labels[B.halfndim:]
        return DiagonalTensor(cData, cLabels)
    elif type(A)==Tensor and type(B)==DiagonalTensor:
        return direct_product(A, diagonalTensor_to_tensor(B))
    elif type(A)==DiagonalTensor and type(B)==Tensor:
        return direct_product(diagonalTensor_to_tensor(A), B)



def contract_indices(A, B, aIndicesContract, bIndicesContract):
    aIndicesContract = A.normarg_indices_back(aIndicesContract)
    bIndicesContract = B.normarg_indices_front(bIndicesContract)
    aLabelsContract = A.labels_of_indices(aIndicesContract)
    bLabelsContract = B.labels_of_indices(bIndicesContract)
    aDimsContract = A.dims(aIndicesContract)
    bDimsContract = B.dims(bIndicesContract)
    if aDimsContract != bDimsContract:
        raise ShapeError(f"{A}, {B}, {aIndicesContract}, {bIndicesContract}, {aDimsContract}, {bDimsContract}")

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
        except (IndicesLengthError, CantKeepDiagonalityError):
            C = direct_product(A, B)
            cIndicesContract1 = [x if x<A.halfndim else x+B.halfndim for x in aIndicesContract]
            cIndicesContract2 = [x+A.halfndim if x<B.halfndim else x+A.ndim for x in bIndicesContract]
            return C.contract_internal_indices(cIndicesContract1, cIndicesContract2)

    elif type(A)==Tensor and type(B)==DiagonalTensor:
        try:
            A = A.move_indices_to_bottom(aIndicesContract)
            aIndicesContract = list(range(A.ndim-len(aIndicesContract), A.ndim))
            aIndicesNotContract = list(range(A.ndim-len(aIndicesContract)))
            B = B.move_half_all_indices_to_top(bIndicesContract)
            bIndicesContract = list(range(len(bIndicesContract)))
            bIndicesNotContract = list(range(len(bIndicesContract), B.ndim))
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
        except (IndicesLengthError, CantKeepDiagonalityError):
            return contract(A, diagonalTensor_to_tensor(B), aIndicesContract, bIndicesContract)

    elif type(A)==DiagonalTensor and type(B)==Tensor:
        return contract(B, A, bIndicesContract, aIndicesContract)

    else:
        return NotImplemented



def contract_common(A, B):
    aLabels = A.labels
    bLabels = B.labels
    commonLabels = intersection_list(aLabels, bLabels)
    return contract_indices(A, B, commonLabels, commonLabels)



def contract(A, B, aIndicesContract=None, bIndicesContract=None):
    if aIndicesContract is None:
        return contract_common(A, B)
    else:
        return contract_indices(A, B, aIndicesContract, bIndicesContract)





