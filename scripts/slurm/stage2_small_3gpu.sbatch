#!/bin/bash
#SBATCH -J sema_stage2_small
#SBATCH -p preempt
#SBATCH --gres=gpu:A40:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o files/logs/%x_%j.out
#
# Submit from the repo root:  sbatch scripts/slurm/stage2_small_3gpu.sbatch
# GPU type is pinned: an any-type request can land on 16 GB T4s, too small for this job.
# Needs OPENAI_API_KEY in .env. 'preempt' jobs can be stopped; checkpoints save every 10 steps.

source ~/.bashrc
conda activate sema_run
export HF_HOME=/scratch/$USER/hf_cache

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
bash scripts/rl_ida_small_3gpu.sh
