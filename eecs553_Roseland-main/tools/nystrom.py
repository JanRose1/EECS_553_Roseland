import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

def Nystrom(data, sample, dim, denoise=False, sig=None, alpha=0):
    """
    Nyström method adapts to Diffusion Maps (DM).
    """

    n = sample.shape[0]

    KNN = n

    nbrs = NearestNeighbors(n_neighbors=KNN, algorithm='auto').fit(sample)
    distance, index = nbrs.kneighbors(sample)

    if sig is None:
        sig = np.quantile(distance[:, -1] ** 2, 0.98)

    ker = np.exp(-3 * (distance ** 2) / sig)
    ii = np.repeat(np.arange(n)[:, np.newaxis], KNN, axis=1)
    A = csr_matrix((ker.ravel(), (ii.ravel(), index.ravel())), shape=(n, n))
    A = A.maximum(A.transpose()).toarray()

    if denoise:
        np.fill_diagonal(A, 0)

    if alpha:
        D = np.power(A.sum(axis=1), 1)
        A = A / D[:, None]
        A = A / D

    D = np.power(A.sum(axis=1), -0.5)
    D = np.diag(D)
    A = D @ A @ D
    A = (A + A.T) / 2
    s, u = eigs(A, k=dim + 1, which='LM')
    u = D @ u
    
    B = np.exp(-cdist(data, sample) ** 2 / sig)
    D_ext = B.sum(axis=1)[:, None]

    u_ext = B @ (u @ np.linalg.inv(np.diag(s))).real / D_ext
    u_ext = np.vstack([u.real, u_ext])
    s = s.real
    return u_ext, s