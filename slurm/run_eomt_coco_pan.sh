#!/bin/bash

#SBATCH --job-name=panseg_coco_eomt_large_640  # job name

#SBATCH --partition=mlgpu_long  #[mlgpu<A40>/sgpu<A100>][devel,short,medium,long]

#SBATCH --time=72:00:00  #HH:MM:SS

#SBATCH --gpus=8

#SBATCH --cpus-per-task=128

#SBATCH --mem=480G

#SBATCH --ntasks=1

#SBATCH -o ./logs/%x.%j.out # where to save the log

GPUS=8
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

source ~/.bashrc
conda activate eomt
module load CUDA/12.6.0

cd ~/eomt

srun python /home/hkuang_hpc/eomt/main.py fit \
  -c /home/hkuang_hpc/eomt/configs/coco/panoptic/eomt_dino_large_672.yaml \
  --trainer.devices 2 \
  --data.batch_size 1 \
  --data.path /lustre/mlnvme/data/hkuang_hpc-hkuang_data/coco 

