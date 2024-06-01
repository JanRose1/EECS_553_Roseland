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

#SBATCH --output=/home/richliu/eecs553_Roseland/output/outputpca.out
#SBATCH --account=eecs553w24_class
#SBATCH --partition=standard

module load python
python3 -m pip install numpy
python3 -m pip install scikit-learn
python3 -m pip install scipy

python main_PCA.py