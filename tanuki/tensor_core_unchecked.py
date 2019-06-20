
    #methods for basic operations
    @inplacable_tensorMixin_method
    def adjoint(self, row_indices, column_indices=None, style="transpose"):
        row_indices, column_indices = self.normarg_complement_indices(row_indices,column_indices)
        if style=="transpose":
            assert len(row_indices) == len(column_indices), f"adjoint arg must be len(row_indices)==len(column_indices). but row_indices=={row_indices}, column_indices=={column_indices}"
            out = self.conjugate()
            out.replace_indices(row_indices+column_indices, column_indices+row_indices)
        elif style=="aster":
            out = self.conjugate()
            out.aster_labels(row_indices+column_indices)
        return out

    adj = adjoint

    @inplacable_tensorMixin_method
    def hermite(self, row_indices, column_indices=None, assume_definite_and_if_negative_then_make_positive=False):
        re = (self + self.adjoint(row_indices,column_indices))/2
        if assume_definite_and_if_negative_then_make_positive:
            if xp.real(re.data.item(0)) < 0:
                re = re * (-1)
        return re

    @inplacable_tensorMixin_method
    def antihermite(self, row_indices, column_indices=None):
        return (self - self.adjoint(row_indices,column_indices))/2






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
    def move_index_to_top(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(0, labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, 0)

    @outofplacable_tensorMixin_method
    def move_index_to_bottom(self, indexMoveFrom):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.append(labelMove)
        self.data = xp.rollaxis(self.data, indexMoveFrom, self.ndim)

    @outofplacable_tensorMixin_method
    def move_index_to_position(self, indexMoveFrom, position):
        indexMoveFrom = self.normarg_index(indexMoveFrom)
        labelMove = self.labels.pop(indexMoveFrom)
        self.labels.insert(position, labelMove)
        if position <= indexMoveFrom:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position)
        else:
            self.data = xp.rollaxis(self.data, indexMoveFrom, position+1)

    @outofplacable_tensorMixin_method
    def move_indices_to_top(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(moveFrom) + self.labels_of_indices(notMoveFrom)
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, moveFrom):
        moveFrom = self.normarg_indices_back(moveFrom)
        moveTo = list(range(self.ndim-len(moveFrom), self.ndim))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom) + self.labels_of_indices(moveFrom)
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, moveFrom, position):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(position, position+len(moveFrom)))
        notMoveFrom = diff_list(list(range(self.ndim)), moveFrom)
        newLabels = self.labels_of_indices(notMoveFrom)
        newLabels = newLabels[:position] + self.labels_of_indices(moveFrom) + newLabels[position:]
        self.data = xp.moveaxis(self.data, moveFrom, moveTo)
        self.labels = newLabels

    @outofplacable_tensorMixin_method
    def move_all_indices(self, moveFrom):
        moveFrom = self.normarg_indices_front(moveFrom)
        moveTo = list(range(self.ndim))
        newLabels = self.labels_of_indices(moveFrom)

        oldPos_NewPos = [None for _ in range(self.ndim)]
        for i in range(self.ndim):
            oldPos_NewPos[moveTo[i]] = moveFrom[i]

        self.data = xp.transpose(self.data, oldPos_NewPos)
        self.labels = newLabels



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
    def pad_indices(self, indices, npads):
        indices = self.normarg_indices(indices)
        wholeNpad = [(0,0)] * self.ndim
        for index, npad in zip(indices, npads):
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
    def contract_internal(self, index1, index2):
        index1 = self.normarg_index_front(index1)
        index2 = self.normarg_index_back(index2)
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
    def trace(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_common_internal()
        else:
            return self.contract_internal(index1, index2)

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

    def to_matrix(self, row_indices, column_indices=None):
        return tensor_to_matrix(self, row_indices, column_indices)

    def to_vector(self, indices):
        return tensor_to_vector(self, indices)

    def to_scalar(self):
        return tensor_to_scalar(self)


    #methods for confirming character
    def is_scalar(self):
        return self.ndim==0

    def is_diagonal(self, absolute_threshold=1e-10):
        return is_diagonal_matrix(self.data, absolute_threshold=absolute_threshold)

    def is_identity(self, absolute_threshold=1e-10):
        if self.ndim != 2:
            return False
        temp = self.data - xp.eye(*self.data.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_prop_identity(self, absolute_threshold=1e-10):
        return is_prop_identity_matrix(self.data, absolute_threshold=absolute_threshold)

    def is_right_unitary(self, column_indices, absolute_threshold=1e-10):
        column_indices, row_indices = self.normarg_complement_indices(column_indices)
        M = self.to_matrix(row_indices, column_indices)
        temp = xp.dot(M, M.conj().transpose())
        temp = temp - xp.eye(*temp.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_left_unitary(self, row_indices, absolute_threshold=1e-10):
        row_indices, column_indices = self.normarg_complement_indices(row_indices)
        M = self.to_matrix(row_indices, column_indices)
        temp = xp.dot(M.conj().transpose(), M)
        temp = temp - xp.eye(*temp.shape)
        return xp.linalg.norm(temp) <= absolute_threshold

    def is_right_prop_unitary(self, column_indices, absolute_threshold=1e-10):
        column_indices, row_indices = self.normarg_complement_indices(column_indices)
        M = self.to_matrix(row_indices, column_indices)
        temp = xp.dot(M, M.conj().transpose())
        return is_prop_identity_matrix(temp, absolute_threshold=absolute_threshold)

    def is_left_prop_unitary(self, row_indices, absolute_threshold=1e-10):
        row_indices, column_indices = self.normarg_complement_indices(row_indices)
        M = self.to_matrix(row_indices, column_indices)
        temp = xp.dot(M.conj().transpose(), M)
        return is_prop_identity_matrix(temp, absolute_threshold=absolute_threshold)

    def is_unitary(self, row_indices, column_indices=None, absolute_threshold=1e-10):
        row_indices, column_indices = self.normarg_complement_indices(row_indices, column_indices)
        if soujou(self.dims_front(row_indices)) != soujou(self.dims_back(column_indices)):
            return False
        M = self.to_matrix(row_indices, column_indices)
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




# A[i,j,k,l] = i==k and j==l and A.data[i,j]
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
        labelsMove = normarg_labels(labelsMove)
        self.labels = labelsMove + diff_list(self.labels, labelsMove)

    @outofplacable_tensorMixin_method
    def move_indices_to_bottom(self, labelsMove):
        labelsMove = normarg_labels(labelsMove)
        self.labels = diff_list(self.labels, labelsMove) + labelsMove

    @outofplacable_tensorMixin_method
    def move_indices_to_position(self, labelsMove, position):
        labelsMove = normarg_labels(labelsMove)
        labelsNotMove = diff_list(self.labels, labelsMove)
        self.labels = labelsNotMove[:position] + labelsMove + labelsNotMove[position:]

    @outofplacable_tensorMixin_method
    def move_all_indices(self, newLabels):
        newLabels = normarg_labels(newLabels)
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

    def __rtruediv__(self, other):
        if isinstance(other, TensorMixin):
            return other * self.inv()
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
        return DiagonalTensor(data=self.data.conj(), labels=self.labels)

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
    # A[i,j,k,l,m,n] = [i==l][j==m][k==n]a[i,j,k]
    # \sum[j==k]A[i,j,k,l,m,n] = \sum_j [i==l][j==m][j==n]a[i,j,j] = [i==l][m==n] \sum_j a[i,j,j]
    # TODO not tested
    @inplacable_tensorMixin_method
    def contract_internal(self, index1, index2):
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
    def contract_common_internal(self):
        temp = self
        commons = floor_half_list(temp.labels)
        for common in commons:
            temp = temp.contract_internal(common, common)
        return temp

    @inplacable_tensorMixin_method
    def trace(self, index1=None, index2=None):
        if index1 is None:
            return self.contract_common_internal()
        else:
            return self.contract_internal(index1, index2)

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

    def to_matrix(self, row_indices, column_indices=None):
        return tensor_to_matrix(self.to_tensor(), row_indices, column_indices)

    def to_vector(self, indices):
        return tensor_to_vector(self.to_tensor(), indices)

    def to_scalar(self):
        return tensor_to_scalar(self.to_tensor())

    def to_diagonalElements(self):
        return diagonalTensor_to_diagonalElements(self)

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
    aLabelsContract = normarg_labels(aLabelsContract)
    bLabelsContract = normarg_labels(bLabelsContract)

    aIndicesContract = aTensor.indices_of_labels_back(aLabelsContract)
    bIndicesContract = bTensor.indices_of_labels_front(bLabelsContract)

    aDimsContract = aTensor.dims(aIndicesContract)
    bDimsContract = bTensor.dims(bIndicesContract)

    assert aDimsContract == bDimsContract: f"{aTensor}, {bTensor}, {aLabelsContract}, {bLabelsContract}"

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
    total_row_dim = soujou(t.shape[:len(row_labels)])
    total_column_dim = soujou(t.shape[len(row_labels):])

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

def diagonalElements_to_diagonalTensor(diagonalElements, labels):
    return DiagonalTensor(diagonalElements, labels)

def diagonalTensor_to_diagonalElements(diagonalTensor):
    return diagonalTensor.data



# iroiro

def is_diagonal_matrix(matrix, absolute_threshold=1e-10):
    if matrix.ndim != 2:
        return False
    temp = matrix[xp.eye(*matrix.shape)==0]
    return xp.linalg.norm(temp) <= absolute_threshold

def is_prop_identity_matrix(matrix, absolute_threshold=1e-10):
    if not is_diagonal_matrix(matrix, absolute_threshold=absolute_threshold):
        print("not diagonal!")
        return False
    d = xp.diag(matrix)
    re = xp.real(d)
    maxre = xp.amax(re)
    minre = xp.amin(re)
    if abs(maxre-minre) > absolute_threshold:
        print("re is bad!")
        return False
    im = xp.imag(d)
    maxim = xp.amax(im)
    minim = xp.amin(im)
    if abs(maxim-minim) > absolute_threshold:
        print("im is bad!")
        return False
    return True