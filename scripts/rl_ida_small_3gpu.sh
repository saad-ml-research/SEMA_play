#!/bin/bash
# Stage II — RL with Intent-drift-aware Reward, small attacker + small victim on 3 GPUs (48 GB class).
#
# Based on rl_ida_qwen3b@llama8b_4x46gb-gpu.sh.
#   GPU A : trl vllm-serve (attacker rollouts for GRPO)
#   GPU B,C : accelerate launch --num_processes 2 (trainer + co-located vLLM victim)
#
# Batch math kept equal to the paper:
#   per_device_train_batch_size 2 x 2 ranks x gradient_accumulation_steps 42 = 168 (global batch)
#   per_device_train_batch_size 2 x 2 ranks x steps_per_generation 84      = 336 (generation batch)
#   336 / num_generations 28 = 12 unique prompts per generation cycle
#
# Changed for a cheap first run: 1 epoch, small victim. Judge is still gpt-4.1-mini (needs OPENAI_API_KEY in .env).
set -e

if [ -f .env ]; then
    set -a; . ./.env; set +a
else
    echo "ERROR: .env not found at repo root"; exit 1
fi
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY is empty in .env"; exit 1
fi

ATTACKER_CKPT="${ATTACKER_CKPT:-files/checkpoints/prefilling_self-tuned-Qwen2.5-0.5B-Instruct}"
VICTIM="${VICTIM:-Qwen/Qwen2.5-1.5B-Instruct}"
EPOCHS="${EPOCHS:-1}"
RUN_TAG="$(basename "$ATTACKER_CKPT" | sed 's/prefilling_self-tuned-//')@$(basename "$VICTIM")"
SERVER_LOGFILE="files/logs/rl_grpo_server_${RUN_TAG}.log"
mkdir -p files/logs

if [ ! -d "$ATTACKER_CKPT" ]; then
    echo "ERROR: Stage I checkpoint not found at $ATTACKER_CKPT — run Stage I first."; exit 1
fi

# Use the GPU ids Slurm gave us (works whether or not Slurm renumbers them from 0).
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [ "${#GPUS[@]}" -lt 3 ]; then
    echo "ERROR: need 3 GPUs, got CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; exit 1
fi
SERVER_GPU="${GPUS[0]}"
TRAIN_GPUS="${GPUS[1]},${GPUS[2]}"
echo "Stage II | attacker=$ATTACKER_CKPT | victim=$VICTIM | epochs=$EPOCHS | server GPU=$SERVER_GPU | train GPUs=$TRAIN_GPUS"

# --- Attacker vLLM server ---
CUDA_VISIBLE_DEVICES="$SERVER_GPU" trl vllm-serve --model "$ATTACKER_CKPT" > "$SERVER_LOGFILE" 2>&1 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
echo "Started vLLM attacker server (PID $VLLM_PID); waiting for http://localhost:8000 ..."
for i in $(seq 1 600); do
    if curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
        echo "vLLM server is healthy after ${i}s."; break
    fi
    sleep 1
done
if ! curl -sSf http://localhost:8000/health >/dev/null 2>&1; then
    echo "ERROR: vLLM server did not become healthy in 600s — see $SERVER_LOGFILE"; exit 1
fi

# --- Trainer (2 ranks) ---
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" accelerate launch --num_processes 2 \
    --config_file scripts/deepspeed_configs/deepspeed_zero3.yaml \
    sema/cli/train_cli.py rl_grpo \
    --seed 42 \
    --data_seed 42 \
    --model_name_or_path "$ATTACKER_CKPT" \
    --output_dir "files/checkpoints/SEMA-${RUN_TAG}" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 42 \
    --steps_per_generation 84 \
    --num_generations 28 \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --torch_dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 500 \
    --use_vllm \
    --vllm_mode server \
    --vllm_server_base_url http://localhost:8000 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --log_completions \
    --report_to wandb \
    --run_name "rl_grpo_${RUN_TAG}" \
    --dataset_name advbench \
    --data_split_seed 42 \
    --test_size 0.2 \
    --max_turns 7 \
    --victim_model_name_or_path "$VICTIM" \
    --gpu_memory_utilization 0.4 \
    --score_reward_name ida_reward \
    --score_reward_model_name openai/gpt-4.1-mini \
    --reward_log_save_dir files/logs

echo "Trainer finished."
