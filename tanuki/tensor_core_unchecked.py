


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