import numpy as np
from sklearn.datasets import fetch_openml
import os
from sklearn.decomposition import PCA
from tools import get_acc
import time
import pandas as pd

current_dir = os.getcwd()

# Constants
#N = 70000
#embed_dim = 50
kmean_iter = 12
Add_noise = [0, 1]

# Constants Roselind
N = 60399
embed_dim = 7


# Load data
#mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
data = (pd.read_csv("Patient1Values.csv")).to_numpy()
data_new = np.array(data[:, 3:10], dtype=float) #feature vectors
label = np.array(data[:,10:11], dtype=int) #last col is the activity label
data_new = data_new[:N]
label = label[:N].flatten()
sorted_indices = label.argsort()
data_new = data_new[sorted_indices]
label = label[sorted_indices]

#data = mnist.data / 255.0
#label = mnist.target.astype(int)
#data = data[:N]
#label = label[:N]
#sorted_indices = label.argsort()
#data = data[sorted_indices]
#label = label[sorted_indices]

for add_noise in Add_noise:
    # Add noise if required
    if add_noise == 1:
        noise = 0.2 * np.random.randn(*data_new.shape)
        # Uncomment below for student t-distribution noise
        # noise = np.random.standard_t(4, size=data.shape) / np.sqrt(2) * 0.2
        data_new += noise

    # Perform PCA
    pca_start_time = time.time()
    pca = PCA(n_components=embed_dim)
    pca_embed = pca.fit_transform(data_new)
    time_pca = time.time() - pca_start_time

    # Calculate accuracy
    acc_pca = get_acc.get_acc(pca_embed, label, kmean_iter)

    # Save the results (adjust 'path' according to your directories)
    if add_noise:
        path = os.path.join(current_dir, 'noisy_data', 'pca')
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
    else:
        path = os.path.join(current_dir, 'clean_data', 'pca')
        os.makedirs(path, exist_ok=True)
        os.chdir(path)

    pca_embed_red = pca_embed[:, :3]
    np.save('label.npy', label)
    np.save('time_pca.npy', time_pca)
    np.save('pca_embed.npy', pca_embed_red)
    np.save('acc_pca.npy', acc_pca)