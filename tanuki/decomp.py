from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki import tensor_core as tnc
import warnings

#decomposition functions
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


def tensor_svd(A, rows, cols=None, svd_labels=2):
    #I believe gesvd and gesdd return s which is positive, descending #TODO check
    #A == U*S*V
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

    mid_dim = s_diag.shape[0]

    U = tnc.matrix_to_tensor(u, row_dims+(mid_dim,), row_labels+[svd_labels[0]])
    S = tnc.diagonalElementsVector_to_diagonalTensor(s_diag, (mid_dim,), [svd_labels[1],svd_labels[2]])
    V = tnc.matrix_to_tensor(v, (mid_dim,)+col_dims, [svd_labels[3]]+col_labels)

    return U, S, V


def truncated_svd(A, rows, cols=None, chi=None, rtol=None, atol=None, svd_labels=2):
    svd_labels = normarg_svd_labels(svd_labels)

    U, S, V = tensor_svd(A, rows, cols, svd_labels=svd_labels)
    s_diag = S.data
    trunc_s_diag = s_diag

    if chi:
        trunc_s_diag = trunc_s_diag[:chi]

    if rtol is None:
        rtol = 1e-40
    if atol is None:
        atol = 1e-60
    threshold = atol + rtol * s_diag[0]
    trunc_s_diag = trunc_s_diag[trunc_s_diag > threshold]

    chi = len(trunc_s_diag)

    S.data = trunc_s_diag
    U = U.truncate_index(svd_labels[0], chi)
    V = V.truncate_index(svd_labels[3], chi)

    return U, S, V



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


def tensor_qr(A, rows, cols=None, qr_labels=1, mode="economic"):
    #A == Q*R
    rows, cols = A.normarg_complement_indices(rows, cols)
    qr_labels = normarg_qr_labels(qr_labels)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)

    a = A.to_matrix(rows, cols)

    q, r = xp.linalg.qr(a, mode=mode)

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


def tensor_lq(A, rows, cols=None, lq_labels=1, mode="economic"):
    #A == L*Q
    rows, cols = A.normarg_complement_indices(rows, cols)
    lq_labels = normarg_lq_labels(lq_labels)

    Q, L = tensor_qr(A, cols, rows, qr_labels=[lq_labels[1],lq_labels[0]], mode=mode)

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
def tensor_eigh(A, rows, cols=None, eigh_labels=1):
    rows, cols = A.normarg_complement_indices(rows, cols)
    eigh_labels = normarg_eigh_labels(eigh_labels)
    row_labels, col_labels = A.labels_of_indices(rows), A.labels_of_indices(cols)
    row_dims, col_dims = A.dims(rows), A.dims(cols)
    dim = soujou(row_dims)
    a = A.to_matrix(rows, cols)
    w,v = xp.linalg.eigh(a)
    V = tnc.matrix_to_tensor(v, row_dims+(dim,), row_labels+[eigh_labels[0]])
    W = tnc.diagonalElementsVector_to_diagonalTensor(w, (dim,), [eigh_labels[1],eigh_labels[2]])
    Vh = tnc.matrix_to_tensor(xp.transpose(xp.conj(v)), (dim,)+col_dims, [eigh_labels[3]]+col_labels)
    return V,W,Vh


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


def truncated_eigh(A, rows, cols=None, chi=None, rtol=None, atol=None, eigh_labels=1):
    eigh_labels = normarg_eigh_labels(eigh_labels)

    V,W,Vh = tensor_eigh(A, rows, cols, eigh_labels=eigh_labels)
    w_diag = W.data
    trunc_w_diag = w_diag

    if chi:
        trunc_w_diag = trunc_w_diag[-chi:]

    if rtol is None:
        rtol = 1e-14
    if atol is None:
        atol = 1e-14
    threshold = atol + rtol * w_diag[-1]
    trunc_w_diag = trunc_w_diag[xp.fabs(trunc_w_diag) > threshold]

    chi = len(trunc_w_diag)

    W.data = trunc_w_diag
    V = V.truncate_index(eigh_labels[0], len(w_diag)-chi, len(w_diag))
    Vh = Vh.truncate_index(eigh_labels[3], len(w_diag)-chi, len(w_diag))

    return V, W, Vh
