import numpy as np
import matplotlib.pyplot as plt
import os

# Define the path to avoid repetition
curr_dir = os.getcwd()
base_path = os.path.join(curr_dir, 'acc_landmarks')

# Load the data
os.chdir(os.path.join(base_path, 'clean_data_clean_landmark'))
acc_roseland_clean_clean = np.load('acc_roseland.npy')

os.chdir(os.path.join(base_path, 'noisy_data_clean_landmark'))
acc_roseland_noisy_clean = np.load('acc_roseland.npy')

os.chdir(os.path.join(base_path, 'noisy_data_noisy_landmark'))
acc_roseland_noisy_noisy = np.load('acc_roseland.npy')

print(acc_roseland_clean_clean.shape)

# Define parameters
Subset_size = np.arange(25, 251, 15)
Embed_dim = [20, 30, 50]

# Fix dim, accuracy vs number of landmark
pick = 0  # in Python, indexing starts at 0
dim_pick = 0

plt.figure(figsize=(9, 5))
plt.plot(Subset_size[pick:], acc_roseland_clean_clean[dim_pick, pick:], '--o', markersize=10, linewidth=3, label='clean data')
plt.plot(Subset_size[pick:], acc_roseland_noisy_noisy[dim_pick, pick:], '--s', markersize=10, linewidth=3, label='noisy data, noisy landmark')
plt.plot(Subset_size[pick:], acc_roseland_noisy_clean[dim_pick, pick:], '--^', markersize=10, linewidth=3, label='noisy data, clean landmark')

plt.grid(which='both', linestyle='--', linewidth=0.5)

if pick == 2:  # Corresponds to the original pick == 3 in MATLAB
    plt.yticks(np.arange(0.6, 0.851, 0.03)) 

Xticks = np.arange(25, 251, 25)
plt.xticks(Xticks[pick:])
plt.gca().tick_params(axis='both', labelsize=15)
#plt.tight_layout()
plt.xlabel('Number of landmarks', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.legend(fontsize=15)

# For saving the figure, we use plt.savefig instead of export_fig
# plt.savefig('acc_landmark_dim20.eps', transparent=True, format='eps')

plt.show()  # To display the figure
