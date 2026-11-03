#!/bin/bash
#SBATCH -J sema_stage1_small
#SBATCH -p preempt
#SBATCH --gres=gpu:L40S:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o files/logs/%x_%j.out
#
# Submit from the repo root:  mkdir -p files/logs && sbatch scripts/slurm/stage1_small_l40s.sbatch
# Note: 'preempt' jobs can be stopped if a higher-priority job needs the node.

source ~/.bashrc
conda activate sema_run
export HF_HOME=/scratch/$USER/hf_cache
# Keep compiler caches off the 20 GB home quota.
export TRITON_CACHE_DIR=/scratch/$USER/cache/triton
export VLLM_CACHE_ROOT=/scratch/$USER/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/scratch/$USER/cache/inductor

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
bash scripts/prefill_selftuning_small_2xL40S.sh
