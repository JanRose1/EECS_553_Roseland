import numpy as np
import os
from random import sample
import time
from tools import roseland, nystrom, get_acc
from sklearn.datasets import fetch_openml

current_dir = os.getcwd()

# Constants
add_noise = 0
clean_subset = 0
N = 70000
iter = 15
kmean_iter = 5
Beta = [0.3, 0.4, 0.5]

# Get Data
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
data = mnist.data / 255.0
label = mnist.target.astype(int)
data = data[:N]
label = label[:N]
sorted_indices = label.argsort()
data = data[sorted_indices]
label = label[sorted_indices]

# Add noise
if add_noise == 1:
    clean_data = data.copy()
    noise = 0.2 * np.random.randn(*data.shape)  # Gaussian
    # noise = np.random.standard_t(4, size=data.shape) / np.sqrt(2) * 0.2  # Student t
    data += noise

# Roseland & Nystrom & accuracy
for beta in Beta:
    subset_size = round(N**beta)
    embed_dim = min(50, subset_size - 1)
    
    acc_roseland = np.zeros((iter, embed_dim))
    acc_nys = np.zeros((iter, embed_dim))
    Time_nys = np.zeros(iter)
    Time_roseland = np.zeros(iter)

    for ii in range(iter):
        # Landmarks
        subind = sample(range(N), N)  # Random permutation of indices
        if clean_subset == 1 and add_noise == 1:
            subset = clean_data[subind[:subset_size]]
        else:
            subset = data[subind[:subset_size]]

        # Roseland
        start_time = time.time()
        roseland_embed, _ = roseland.Roseland(data, embed_dim, subset, 1)
        time_roseland = time.time() - start_time
        Time_roseland[ii] = time_roseland

        # Nystrom
        start_time = time.time()
        nystrom_embed, _ = nystrom.Nystrom(data, subset, embed_dim, 0)
        time_nys = time.time() - start_time
        Time_nys[ii] = time_nys
        nystrom_embed = nystrom_embed[subset_size:, 1:]

        # Accuracy
        acc_roseland[ii, :] = get_acc.get_acc(roseland_embed, label, kmean_iter)
        acc_nys[ii, :] = get_acc.get_acc(nystrom_embed, label, kmean_iter)
    
    # Save
    path = os.path.join(current_dir, 'clean_data', f'beta{beta:.1f}')
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    roseland_embed = roseland_embed[:, :3]
    nystrom_embed = nystrom_embed[:, :3]

    np.save('label.npy', label)
    np.save('roseland_embed.npy', roseland_embed)
    np.save('nystrom_embed.npy', nystrom_embed)
    np.save('acc_roseland.npy', acc_roseland)
    np.save('acc_nys.npy', acc_nys)
    np.save('time_nys.npy', Time_nys)
    np.save('time_roseland.npy', Time_roseland)