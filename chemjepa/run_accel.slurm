#!/bin/bash
#SBATCH --nodes=1
##SBATCH --cpus-per-task=128
##SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#SBATCH --partition g5-12xlarge
#conda activate work39
export HF_DATASETS_CACHE=/shared/.cache/huggingface/dataset
srun conda run -n mfdoom --no-capture-output accelerate launch $1 $2
