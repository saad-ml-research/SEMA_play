#!/bin/bash
# Stage I — Prefilling Self-Tuning with a small attacker, for 2 x L40S (48 GB).
#
# Same recipe as prefill_selftuning_qwen3b_4x48gb-gpu.sh, but with a small model
# and 2 training processes. Stage I does not call the OpenAI judge.
#
# Override the model with:  MODEL=Qwen/Qwen3-0.6B bash scripts/prefill_selftuning_small_2xL40S.sh
# (Default is Qwen2.5-0.5B-Instruct: same family as the repo's Qwen-3B config and
#  no "thinking" tokens, which would otherwise end up in the self-tuning data.)
set -e

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TAG="$(basename "$MODEL")"
ROLLOUT_FILE="${TAG}_prefilling_rollout.json"
NUM_GPUS="${NUM_GPUS:-2}"

echo "Stage I | model=$MODEL | gpus=$NUM_GPUS"

python -m sema.cli.train_cli rollout \
    --seed 42 \
    --model_name_or_path "$MODEL" \
    --temperature 1.0 \
    --max_tokens 500 \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --n_generations 10 \
    --system_prompt_path sema/prompts/sema_system.jinja \
    --query_prompt_path sema/prompts/sema_query.jinja \
    --batch_size 100 \
    --output_dir files/prefilling_rollouts \
    --output_file "$ROLLOUT_FILE"

accelerate launch --num_processes "$NUM_GPUS" \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py self_tuning \
    --torch_dtype bfloat16 \
    --model_name_or_path "$MODEL" \
    --output_dir "files/checkpoints/prefilling_self-tuned-${TAG}" \
    --dataset_name json \
    --data_path "files/prefilling_rollouts/${ROLLOUT_FILE}" \
    --gradient_checkpointing \
    --max_length 4096 \
    --per_device_train_batch_size 4 \
    --num_train_epochs 1 \
    --eval_strategy "steps" \
    --eval_steps 100 \
    --logging_steps 1 \
    --save_strategy no \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --report_to wandb \
    --run_name "prefilling_self-tuning_${TAG}" \
    --seed 42
