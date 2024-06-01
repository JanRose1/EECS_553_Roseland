import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def Roseland(data, dim, ref, denoise):
    n = data.shape[0]

    # form affinity matrix wrt the ref set
    affinity_ext = cdist(data, ref, metric='sqeuclidean')

    # affinity matrix W
    sig = np.median(np.median(affinity_ext, axis=1))

    if denoise:
        W_ref = np.exp(-6 * affinity_ext / sig)
    else:
        W_ref = np.exp(-5 * affinity_ext / sig)
    
    W_ref = csr_matrix(W_ref)
    # make W row stochastic
    D = W_ref.dot(np.sum(W_ref, axis=0).T)
    D = np.sqrt(np.linalg.pinv(D))

    # form sparse D = D^(-.5)
    D = csr_matrix((D.A.flatten(), (np.arange(n), np.arange(n))), shape=(n, n))
    W_ref = D.dot(W_ref)

    # SVD on D * W_ref
    k = min(min(W_ref.shape), dim+1)
    U, S, _ = svds(W_ref, k=k, solver='propack')
    U = D.dot(U[:, 1:])
    S = S[1:]**2

    return U, S