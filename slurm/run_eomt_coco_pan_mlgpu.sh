#!/bin/bash

#SBATCH --job-name=panseg_coco_eomt_dinov3_640  # job name

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

export PYTHONPATH=/home/hkuang_hpc/eomt/models:$PYTHONPATH
export PYTHONPATH=/home/hkuang_hpc/eomt/models/dinov3:$PYTHONPATH

cd ~/eomt

srun python main.py fit \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 8 \
  --data.batch_size 2 \
  --data.path /lustre/mlnvme/data/hkuang_hpc-hkuang_data/coco \
  --model.init_args.lr 1e-5
