#!/bin/bash

#SBATCH --job-name=final_roseland
#SBATCH --mail-user=richliu@umich.edu
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --export=ALL

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=8:00:00

#SBATCH --output=/home/richliu/eecs553_Roseland/output/output.out
#SBATCH --account=eecs553w24_class
#SBATCH --partition=standard

module load python
python3 -m pip install numpy
python3 -m pip install scikit-learn
python3 -m pip install scipy

python clean_main_nys_roseland.py
echo 1
python noisy_main_nys_roseland.py
echo 2
python main_PCA.py
echo 3
python main_DM.py
echo 4
python clean_main_acc_landmarks.py
echo 5
python noisy_cleansub_main_acc_landmarks.py
echo 6
python noisy_noisysub_main_acc_landmarks.py
echo 7