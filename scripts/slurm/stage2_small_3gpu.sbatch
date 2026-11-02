#!/bin/bash
#SBATCH -J sema_stage2_small
#SBATCH -p preempt
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o files/logs/%x_%j.out
#
# Submit from the repo root:  sbatch scripts/slurm/stage2_small_3gpu.sbatch
# Needs OPENAI_API_KEY in .env. 'preempt' jobs can be stopped; checkpoints save every 10 steps.

source ~/.bashrc
conda activate sema_run
export HF_HOME=/scratch/$USER/hf_cache

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
bash scripts/rl_ida_small_3gpu.sh
