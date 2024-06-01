import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tools import diffusionmap, get_acc
import time
import os

current_dir = os.getcwd()

# Constants
Add_noise = [0, 1]
N = 70000
embed_dim = 50
kmean_iter = 12

# Get data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
data = mnist.data / 255.0
label = mnist.target.astype(int)
data = data[:N]
label = label[:N]
sorted_indices = label.argsort()
data = data[sorted_indices]
label = label[sorted_indices]

# Add noise
for add_noise in Add_noise:
    if add_noise:
        clean_data = data.copy()
        noise = 0.2 * np.random.randn(*data.shape)  # Gaussian noise
        # noise = np.random.standard_t(4, size=data.shape) / np.sqrt(2) * 0.2  # Student-t noise
        data += noise

    # DM & accuracy
    start_time = time.time()
    dm_embed, _ = diffusionmap.Diffusion_Map(data, embed_dim, 200, False)
    time_dm = time.time() - start_time

    # Calculate accuracy
    acc_dm = get_acc.get_acc(dm_embed, label, kmean_iter)

    # Save files
    dm_embed = dm_embed[:, :3]
    if add_noise:
        path = os.path.join(current_dir, 'noisy_data', 'dm')
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
    else:
        path = os.path.join(current_dir, 'clean_data', 'dm')
        os.makedirs(path, exist_ok=True)
        os.chdir(path)

    np.save('label.npy', label)
    np.save('dm_embed.npy', dm_embed)
    np.save('acc_dm.npy', acc_dm)
    np.save('time_dm.npy', time_dm)