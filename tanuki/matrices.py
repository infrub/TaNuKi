import scipy #because cupy is not suitable to treat such little matrices


def zeros(dim):
    return scipy.zeros((dim,dim))

def identity(dim):
    return scipy.identity(dim)


# Pauli spin 1/2 operators:
sigma0 = scipy.array([[0., 0.], [0., 0.]])
sigmai = scipy.array([[1., 0.], [0., 1.]])
sigmap = scipy.array([[0., 1.], [0., 0.]])
sigmam = scipy.array([[0., 0.], [1., 0.]])
sigmax = scipy.array([[0., 1.], [1., 0.]])
sigmay = scipy.array([[0., -1.j], [1.j, 0.]])
sigmaz = scipy.array([[1., 0.], [0., -1.]])


#creation-and-annihilation:
def annihilation(dim):
    return scipy.diag(scipy.sqrt(scipy.arange(1,dim)), 1)

def creation(dim):
    return scipy.diag(scipy.sqrt(scipy.arange(1,dim)), -1)


#useful as a terminal tensor
def basis(dim, i):
    vec = scipy.zeros(dim)
    vec[i] = 1.0
    return scipy.array(vec)