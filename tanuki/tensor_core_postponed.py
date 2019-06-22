
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






class DiagonalTensor(TensorMixin):


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


