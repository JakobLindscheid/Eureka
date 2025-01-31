#!/bin/bash -l
#SBATCH --job-name=eureka_human
#SBATCH --output=outputs/alice/%x_%j.out
#SBATCH --mail-user="jakob.lindscheid@gmail.com"
#SBATCH --mail-type="NONE"
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2080_ti:4

module load ALICE/default
module load Miniconda3/23.9.0-0
module load CUDA/11.8.0

conda activate eureka

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s3942716/.conda/envs/eureka/lib

# Run the file
echo "[$SHELL] ## Run script"
python eureka.py env_parent=isaac env=$1 evaluate_only=True eval_human=True use_wandb=True wandb_name=human_$1_$SLURM_JOB_ID
echo "[$SHELL] ## Script finished"

# sbatch --export=WANDB_API_KEY alice_human.slurm ant
# watch squeue --me --start --state=ALL
