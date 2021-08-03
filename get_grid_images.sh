#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=5G
#SBATCH --time=00:05:00
#SBATCH --job-name=stylegan
#SBATCH --mail-user=ethan.davis@uvm.edu
export PATH=/gpfs3/arch/x86_64-rhel7/cuda-10.0/bin:${PATH}
export LD_LIBRARY_PATH=/gpfs3/arch/x86_64-rhel7/cuda-10.0/lib64:${LD_LIBRARY_PATH}
folder=$1
file=$2
python get_grid_images.py -f ${folder} -i ${file}

