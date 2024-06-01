import numpy as np
from sklearn.cluster import KMeans

def get_acc(embedding, label, K):
    """
    embedding: data embeddings (numpy.array)
    label: true labels (numpy.array)
    K: number of k-means to average (int)
    """
    dim = embedding.shape[1]
    acc_per_k_dim = []

    for k in range(K):
        acc_per_dim = []
        for d in range(dim):
            kmeans = KMeans(n_clusters=10, random_state=k)
            clusters = kmeans.fit_predict(embedding[:, :d+1])

            acc_per_cluster = []
            for i in range(10):
                cluster_labels = label[clusters == i]
                # Handle case where no points are assigned to a cluster
                if cluster_labels.size == 0:
                    acc = 0.0
                else:
                    pred_cluster_label = np.bincount(cluster_labels).argmax()
                    acc = np.mean(cluster_labels == pred_cluster_label)
                acc_per_cluster.append(acc)
            
            acc_per_dim.append(np.mean(acc_per_cluster))
        acc_per_k_dim.append(acc_per_dim)
    
    acc_per_k_dim = np.array(acc_per_k_dim)
    acc_average = np.mean(acc_per_k_dim, axis=0)
    return acc_average