import numpy as np
import os
from random import sample
from tools import roseland, get_acc2
from sklearn.datasets import fetch_openml

current_dir = os.getcwd()

# Constants
add_noise = True
clean_subset = False
N = 70000
iter = 30
kmean_iter = 5
Subset_size = list(range(25, 251, 15))
Embed_dim = [20, 30, 50]
acc_roseland = np.zeros((len(Embed_dim), len(Subset_size)))

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
if add_noise:
    clean_data = data.copy()
    noise = 0.2 * np.random.randn(*data.shape)  # Gaussian noise
    data += noise

# Roseland & accuracy
for j, subset_size in enumerate(Subset_size):
    acc_roseland_temp = np.zeros((len(Embed_dim), iter))
    for ii in range(iter):
        # Landmarks
        subind = sample(range(N), N)  # Random permutation of indices
        if clean_subset == 1 and add_noise == 1:
            subset = clean_data[subind[:subset_size]]
        else:
            subset = data[subind[:subset_size]]

        # Roseland
        roseland_embed, _ = roseland.Roseland(data, min(subset_size, 50), subset, 1)

        # Accuracy
        accs = get_acc2.get_acc2(roseland_embed, Embed_dim, label, kmean_iter)
        acc_roseland_temp[:, ii] = accs

    acc_roseland[:, j] = np.mean(acc_roseland_temp, axis=1)

# Save
paths = {
    (False, None): 'clean_data_clean_landmark',
    (True, False): 'noisy_data_noisy_landmark',
    (True, True): 'noisy_data_clean_landmark'
}
selected_path = paths[(add_noise, clean_subset if add_noise else None)]
path = os.path.join(current_dir, 'acc_landmarks', selected_path)
os.makedirs(path, exist_ok=True)

# Saving file
np.save(os.path.join(path, 'acc_roseland.npy'), acc_roseland)