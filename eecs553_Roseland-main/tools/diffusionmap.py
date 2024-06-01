import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from sklearn.neighbors import NearestNeighbors

def Diffusion_Map(data, dim, knn, denoise=False):
    n = data.shape[0]

    # Search for K nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=knn, algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Make sure d(x,x)=0
    distances[:, 0] = 0

    # Find sigma
    sig = np.percentile(distances[:, -1] ** 2, 50)
    
    # Affinity matrix
    ker = np.exp(-3.5 * (distances ** 2) / sig)

    ii = np.repeat(np.arange(n)[:, np.newaxis], knn, axis=1)
    W = csr_matrix((ker.ravel(), (ii.ravel(), indices.ravel())), shape=(n, n))
    W = W.maximum(W.transpose())

    if denoise:
        W.setdiag(0)
        W.eliminate_zeros()

    # Graph Laplacian
    D = np.sqrt(W.sum(axis=1).A1)  # Convert sparse matrix sum to 1D numpy array
    D_inv = np.linalg.pinv(D)
    W = W.multiply(D_inv[:, np.newaxis]).multiply(D_inv)

    # Ensure symmetry
    W = (W + W.T) / 2

    # Eigenvalue Decomposition
    S, U = eigs(W, k=dim + 1, which='LM')
    U = U[:, 1:].real
    S = np.diag(S[1:].real)
    U = U / D[:, np.newaxis]

    return U, S
