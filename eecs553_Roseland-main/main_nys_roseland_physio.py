import numpy as np
import pandas as pd
import os
from random import sample
import time
from tools import roseland, nystrom, get_acc3


current_dir = os.getcwd()
add_noise = 0
clean_subset = 0
N = 60399
iter = 15
kmean_iter = 5
Beta = [0.3, 0.4, 0.5]

data = (pd.read_csv('Patient1Values.csv')).to_numpy() #this should be the data after processing it
data_new = np.array(data[:, 3:10], dtype=float) #feature vectors
label = np.array(data[:,10:11], dtype=int) #last col is the activity label
data_new = data_new[:N]
label = label[:N].flatten()

sorted_indices = label.argsort()
data_new = data_new[sorted_indices]
label = label[sorted_indices]

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
        subset = data_new[subind[:subset_size]]

        # Roseland
        start_time = time.time()
        roseland_embed, _ = roseland.Roseland(data_new, embed_dim, subset, 1)
        time_roseland = time.time() - start_time
        Time_roseland[ii] = time_roseland

        # Nystrom
        start_time = time.time()
        nystrom_embed, _ = nystrom.Nystrom(data_new, subset, embed_dim, 0)
        time_nys = time.time() - start_time
        Time_nys[ii] = time_nys
        nystrom_embed = nystrom_embed[subset_size:, 1:]

        # Accuracy
        acc_roseland[ii, :] = get_acc3.get_acc3(roseland_embed, label, kmean_iter)
        acc_nys[ii, :] = get_acc3.get_acc3(nystrom_embed, label, kmean_iter)

    path = os.path.join(current_dir, 'new_dataset', f'beta{beta:.1f}')
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    roseland_embed = roseland_embed[:, :3]
    nystrom_embed = nystrom_embed[:, :3]

    np.save('label_physio.npy', label)
    np.save('roseland_embed_physio.npy', roseland_embed)
    np.save('nystrom_embed_physio.npy', nystrom_embed)
    np.save('acc_roseland_physio.npy', acc_roseland)
    np.save('acc_nys_physio.npy', acc_nys)
    np.save('time_nys_physio.npy', Time_nys)
    np.save('time_roseland_physio.npy', Time_roseland)