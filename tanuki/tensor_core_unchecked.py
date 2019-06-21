

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
    return Tensor(xp.diagflat(diagonalTensor.data), diagonalTensor.labels)

def tensor_to_diagonalTensor(tensor):
    return DiagonalTensor(xp.diagonal(tensor.data), tensor.labels)

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
    d = xp.diagonal(matrix)
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