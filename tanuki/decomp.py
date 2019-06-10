from tanuki.tnxp import xp
from tanuki.utils import *
from tanuki.tensor_core import *

#decomposition functions
def normalize_argument_svd_labels(svd_labels):
    if svd_labels is None:
        svd_labels = [unique_label()]
    if not isinstance(svd_labels, list):
        svd_labels = [svd_labels]
    if len(svd_labels)==1:
        svd_labels = [svd_labels[0]+"_us", svd_labels[0]+"_sv"]
    if len(svd_labels)==2:
        svd_labels = [svd_labels[0],svd_labels[0],svd_labels[1],svd_labels[1]]
    if len(svd_labels)!=4:
        raise ValueError(f"svd_labels must be a None or str or 1,2,4 length list. svd_labels=={svd_labels}")
    return svd_labels


def tensor_svd(A, row_labels, column_labels=None, svd_labels=None):
    #I believe gesvd and gesdd return s which is positive, descending #TODO check
    #A == U*S*V
    row_labels, column_labels = normalize_and_complement_argument_labels(A.labels, row_labels, column_labels)

    svd_labels = normalize_argument_svd_labels(svd_labels)

    row_dims = A.dims_of_labels_front(row_labels)
    column_dims = A.dims_of_labels_back(column_labels)

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

    U = matrix_to_tensor(u, row_dims+(mid_dim,), row_labels+[svd_labels[0]])
    S = diagonalMatrix_to_diagonalTensor(s_diag, [svd_labels[1],svd_labels[2]])
    V = matrix_to_tensor(v, (mid_dim,)+column_dims, [svd_labels[3]]+column_labels)

    return U, S, V


def truncated_svd(A, row_labels, column_labels=None, chi=None, absolute_threshold=None, relative_threshold=None, svd_labels=None):
    svd_labels = normalize_argument_svd_labels(svd_labels)
    U, S, V = tensor_svd(A, row_labels, column_labels, svd_labels=svd_labels)
    s_diag = S.data

    trunc_s_diag = s_diag

    if chi:
        trunc_s_diag = trunc_s_diag[:chi]

    if absolute_threshold:
        trunc_s_diag = trunc_s_diag[trunc_s_diag > absolute_threshold]

    if relative_threshold:
        threshold = relative_threshold * s_diag[0]
        trunc_s_diag = trunc_s_diag[trunc_s_diag > threshold]

    chi = len(trunc_s_diag)

    S.data = trunc_s_diag
    U.move_indices_to_top(svd_labels[0])
    U.data = U.data[0:chi]
    U.move_indices_to_bottom(svd_labels[0])
    V.data = V.data[0:chi]

    return U, S, V



def normalize_argument_qr_labels(qr_labels):
    if qr_labels is None:
        qr_labels = [unique_label()]
    if not isinstance(qr_labels, list):
        qr_labels = [qr_labels]
    if len(qr_labels)==1:
        qr_labels = [qr_labels[0]+"_qr", qr_labels[0]+"_qr"]
    if len(qr_labels)!=2:
        raise ValueError(f"qr_labels must be a None or str or 1,2 length list. qr_labels=={qr_labels}")
    return qr_labels


def tensor_qr(A, row_labels, column_labels=None, qr_labels=None, mode="economic"):
    #A == Q*R
    row_labels, column_labels = normalize_and_complement_argument_labels(A.labels, row_labels, column_labels)
    qr_labels = normalize_argument_qr_labels(qr_labels)

    row_dims = A.dims_of_labels_front(row_labels)
    column_dims = A.dims_of_labels_back(column_labels)

    a = A.to_matrix(row_labels, column_labels)

    q, r = xp.linalg.qr(a, mode=mode)

    mid_dim = r.shape[0]

    Q = matrix_to_tensor(q, row_dims+(mid_dim,), row_labels+[qr_labels[0]])
    R = matrix_to_tensor(r, (mid_dim,)+column_dims, [qr_labels[1]]+column_labels)

    return Q, R


def normalize_argument_lq_labels(lq_labels):
    if lq_labels is None:
        lq_labels = [unique_label()]
    if not isinstance(lq_labels, list):
        lq_labels = [lq_labels]
    if len(lq_labels)==1:
        lq_labels = [lq_labels[0]+"_lq", lq_labels[0]+"_lq"]
    if len(lq_labels)!=2:
        raise ValueError(f"lq_labels must be a None or str or 1,2 length list. lq_labels=={lq_labels}")
    return lq_labels


def tensor_lq(A, row_labels, column_labels=None, lq_labels=None, mode="economic"):
    #A == L*Q
    row_labels, column_labels = normalize_and_complement_argument_labels(A.labels, row_labels, column_labels)
    lq_labels = normalize_argument_lq_labels(lq_labels)

    Q, L = tensor_qr(A, column_labels, row_labels, qr_labels=[lq_labels[1],lq_labels[0]], mode=mode)

    return L, Q



def normalize_argument_eigh_labels(eigh_labels):
    if eigh_labels is None:
        eigh_labels = [unique_label()]
    if not isinstance(eigh_labels, list):
        eigh_labels = [eigh_labels]
    if len(eigh_labels)==1:
        eigh_labels = [eigh_labels[0]+"_eigh", eigh_labels[0]+"_eigh"]
    if len(eigh_labels)!=2:
        raise ValueError(f"eigh_labels must be a None or str or 1,2 length list. eigh_labels=={eigh_labels}")
    return eigh_labels

def tensor_eigh(A, row_labels, column_labels=None)
    row_labels, column_labels = normalize_and_complement_argument_labels(A.labels, row_labels, column_labels)