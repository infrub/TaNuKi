from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki import tensor_core as tnc
import warnings

#decomposition functions
def normarg_qr_labels(qr_labels):
    if qr_labels is None:
        qr_labels = [unique_label()]
    if isinstance(qr_labels, int):
        qr_labels = [unique_label() for _ in range(qr_labels)]
    if not isinstance(qr_labels, list):
        qr_labels = [qr_labels]
    if len(qr_labels)==1:
        qr_labels = [qr_labels[0], qr_labels[0]]
    if len(qr_labels)!=2:
        raise ValueError(f"qr_labels must be a None or str or 1,2 length list. qr_labels=={qr_labels}")
    return qr_labels


def tensor_qr(A, rows, cols=None, qr_labels=1, mode="economic", force_diagonal_elements_positive=False):
    #A == Q*R
    rows, cols = A.normarg_complement_indices(rows, cols)
    qr_labels = normarg_qr_labels(qr_labels)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)

    a = A.to_matrix(rows, cols)

    q, r = xp.linalg.qr(a, mode=mode)

    if force_diagonal_elements_positive:
        d = r.diagonal()
        d = d.real >= 0
        d = d*2-1

        q = q * d
        r = r.transpose()
        r = r * d
        r = r.transpose()

    mid_dim = r.shape[0]

    Q = tnc.matrix_to_tensor(q, row_dims+(mid_dim,), row_labels+[qr_labels[0]])
    R = tnc.matrix_to_tensor(r, (mid_dim,)+col_dims, [qr_labels[1]]+col_labels)

    return Q, R


def normarg_lq_labels(lq_labels):
    if lq_labels is None:
        lq_labels = [unique_label()]
    if isinstance(lq_labels, int):
        lq_labels = [unique_label() for _ in range(lq_labels)]
    if not isinstance(lq_labels, list):
        lq_labels = [lq_labels]
    if len(lq_labels)==1:
        lq_labels = [lq_labels[0], lq_labels[0]]
    if len(lq_labels)!=2:
        raise ValueError(f"lq_labels must be a None or str or 1,2 length list. lq_labels=={lq_labels}")
    return lq_labels


def tensor_lq(A, rows, cols=None, lq_labels=1, mode="economic", force_diagonal_elements_positive=False):
    #A == L*Q
    rows, cols = A.normarg_complement_indices(rows, cols)
    lq_labels = normarg_lq_labels(lq_labels)

    Q, L = tensor_qr(A, cols, rows, qr_labels=[lq_labels[1],lq_labels[0]], mode=mode, force_diagonal_elements_positive=force_diagonal_elements_positive)

    return L, Q



def normarg_eigh_labels(eigh_labels):
    if eigh_labels is None:
        eigh_labels = [unique_label()]
    if isinstance(eigh_labels, int):
        eigh_labels = [unique_label() for _ in range(eigh_labels)]
    if not isinstance(eigh_labels, list):
        eigh_labels = [eigh_labels]
    if len(eigh_labels)==1:
        eigh_labels = [eigh_labels[0], eigh_labels[0]]
    if len(eigh_labels)==2:
        eigh_labels = [eigh_labels[0],eigh_labels[0],eigh_labels[1],eigh_labels[1]]
    if len(eigh_labels)!=4:
        raise ValueError(f"eigh_labels must be a None or str or 1,2,4 length list. eigh_labels=={eigh_labels}")
    return eigh_labels


# A == V*W*Vh
def tensor_eigh(A, rows, cols=None, rtol=1e-30, atol=1e-50, eigh_labels=1):
    rows, cols = A.normarg_complement_indices(rows, cols)
    eigh_labels = normarg_eigh_labels(eigh_labels)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)
    dim = soujou(row_dims)
    a = A.to_matrix(rows, cols)
    w_diag,v = xp.linalg.eigh(a)

    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0
    threshold = atol + rtol * w_diag[-1]
    selector = xp.fabs(w_diag) >= threshold
    w_diag = w_diag[selector]
    v = v[:,selector]

    chi = len(w_diag)

    V = tnc.matrix_to_tensor(v, row_dims+(chi,), row_labels+[eigh_labels[0]])
    W = tnc.diagonalElementsVector_to_diagonalTensor(w_diag, (chi,), [eigh_labels[1],eigh_labels[2]])
    Vh = tnc.matrix_to_tensor(xp.transpose(xp.conj(v)), (chi,)+col_dims, [eigh_labels[3]]+col_labels)

    return V, W, Vh



def normarg_svd_labels(svd_labels):
    if svd_labels is None:
        svd_labels = [unique_label()]
    if isinstance(svd_labels, int):
        svd_labels = [unique_label() for _ in range(svd_labels)]
    if not isinstance(svd_labels, list):
        svd_labels = [svd_labels]
    if len(svd_labels)==1:
        svd_labels = [svd_labels[0], svd_labels[0]]
    if len(svd_labels)==2:
        svd_labels = [svd_labels[0],svd_labels[0],svd_labels[1],svd_labels[1]]
    if len(svd_labels)!=4:
        raise ValueError(f"svd_labels must be a None or str or 1,2,4 length list. svd_labels=={svd_labels}")
    return svd_labels


def tensor_svd(A, rows, cols=None, chi=None, rtol=1e-30, atol=1e-50, svd_labels=2):
    rows, cols = A.normarg_complement_indices(rows, cols)
    svd_labels = normarg_svd_labels(svd_labels)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)

    a = A.to_matrix(rows, cols)

    try:
        u, s_diag, v = xp.linalg.svd(a, full_matrices=False)
    except (xp.linalg.LinAlgError, ValueError):
        warnings.warn("xp.linalg.svd failed with gesdd. retry with gesvd.")
        try:
            u, s_diag, v = xp.linalg.svd(a, full_matrices=False, lapack_driver="gesvd")
        except ValueError:
            raise

    if chi:
        s_diag = s_diag[:chi]

    if rtol is None:
        rtol = 0.0
    if atol is None:
        atol = 0.0
    threshold = atol + rtol * s_diag[0]
    s_diag = s_diag[s_diag >= threshold]

    chi = len(s_diag)
    u = xp.split(u, [0, chi], axis=1)[1]
    v = xp.split(v, [0, chi], axis=0)[1]

    U = tnc.matrix_to_tensor(u, row_dims+(chi,), row_labels+[svd_labels[0]])
    S = tnc.diagonalElementsVector_to_diagonalTensor(s_diag, (chi,), [svd_labels[1],svd_labels[2]])
    V = tnc.matrix_to_tensor(v, (chi,)+col_dims, [svd_labels[3]]+col_labels)

    return U, S, V


# A*V == w*V
def tensor_eigsh(A, rows, cols=None):
    rows, cols = A.normarg_complement_indices(rows, cols)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)
    a = A.to_matrix(rows, cols)
    w,v = xp.sparse.linalg.eigsh(a, k=1)
    w = w[0]
    v = v[:,0]
    V = tnc.vector_to_tensor(v, col_dims, col_labels)
    return w,V



# A*X == B
def tensor_solve(A, B, rows_of_A=None, cols_of_A=None, rows_of_B=None, cols_of_B=None, assume_a="gen"):
    if rows_of_A is None and rows_of_B is None:
        row_labels = intersection_list(A.labels, B.labels)
        rows_of_A,cols_of_A = A.normarg_complement_indices(row_labels, cols_of_A)
        rows_of_B,cols_of_B = B.normarg_complement_indices(row_labels, cols_of_B)
    elif rows_of_A is None:
        rows_of_B, cols_of_B = B.normarg_complement_indices(rows_of_B, cols_of_B)
        row_labels = B.labels_of_indices(rows_of_B)
        rows_of_A,cols_of_A = A.normarg_complement_indices(row_labels, cols_of_A)
    elif rows_of_B is None:
        rows_of_A, cols_of_A = A.normarg_complement_indices(rows_of_A, cols_of_A)
        row_labels = A.labels_of_indices(rows_of_A)
        rows_of_B,cols_of_B = B.normarg_complement_indices(row_labels, cols_of_B)
    else:
        rows_of_A, cols_of_A = A.normarg_complement_indices(rows_of_A, cols_of_A)
        rows_of_B, cols_of_B = B.normarg_complement_indices(rows_of_B, cols_of_B)
    labels_of_X = A.labels_of_indices(cols_of_A) + B.labels_of_indices(cols_of_B)
    shape_of_X = A.dims(cols_of_A) + B.dims(cols_of_B)

    Adata = A.to_matrix(rows_of_A, cols_of_A)
    Bdata = B.to_matrix(rows_of_B, cols_of_B)
    Xdata = xp.linalg.solve(Adata, Bdata, assume_a=assume_a)
    X = tnc.matrix_to_tensor(Xdata, shape_of_X, labels_of_X)

    return X