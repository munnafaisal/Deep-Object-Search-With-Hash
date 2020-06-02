import sys
import numpy
import scipy
from scipy.sparse import csr_matrix, isspmatrix

def numpy_array_from_list_or_numpy_array(vectors):
    """
    Returns numpy array representation of argument.
    Argument maybe numpy array (input is returned)
    or a list of numpy vectors.
    """
    # If vectors is not a numpy matrix, create one
    if not isinstance(vectors, numpy.ndarray):
        V = numpy.zeros((vectors[0].shape[0], len(vectors)))
        for index in range(len(vectors)):
            vector = vectors[index]
            V[:, index] = vector
        return V

    return vectors


def unitvec(vec):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    """
    if scipy.sparse.issparse(vec): # convert scipy.sparse to standard numpy array
        vec = vec.tocsr()
        veclen = numpy.sqrt(numpy.sum(vec.data ** 2))
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec

    if isinstance(vec, numpy.ndarray):
        vec = numpy.asarray(vec, dtype=float)
        veclen = numpy.linalg.norm(vec)
        if veclen > 0.0:
            return vec / veclen
        else:
            return vec


def perform_pca(A):
    """
    Computes eigenvalues and eigenvectors of covariance matrix of A.
    The rows of a correspond to observations, the columns to variables.
    """
    # First subtract the mean
    M = (A-numpy.mean(A.T, axis=1)).T
    # Get eigenvectors and values of covariance matrix
    return numpy.linalg.eig(numpy.cov(M))


PY2 = sys.version_info[0] == 2
if PY2:
    bytes_type = str
else:
    bytes_type = bytes


def want_string(arg, encoding='utf-8'):
    if isinstance(arg, bytes_type):
        rv = arg.decode(encoding)
    else:
        rv = arg
    return rv