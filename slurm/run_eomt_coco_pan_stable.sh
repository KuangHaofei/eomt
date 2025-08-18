#!/bin/bash

#SBATCH --job-name=panseg_coco_eomt_dinov3_640  # job name

#SBATCH --partition=sgpu_long  #[mlgpu<A40>/sgpu<A100>][devel,short,medium,long]

#SBATCH --time=72:00:00  #HH:MM:SS

#SBATCH --gpus=4

#SBATCH --cpus-per-task=128

#SBATCH --mem=480G

#SBATCH --ntasks=1

#SBATCH -o ./logs/%x.%j.out # where to save the log

GPUS=4
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

source ~/.bashrc
conda activate eomt
module load CUDA/12.6.0

export PYTHONPATH=/home/hkuang_hpc/eomt/models:$PYTHONPATH
export PYTHONPATH=/home/hkuang_hpc/eomt/models/dinov3:$PYTHONPATH

cd ~/eomt

srun python main.py fit \
  -c configs/coco/panoptic/eomt_large_640_stable.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /lustre/mlnvme/data/hkuang_hpc-hkuang_data/coco \
  --trainer.gradient_clip_val 1.0 \
  --trainer.precision "16-mixed" \
  --trainer.check_val_every_n_epoch 1
