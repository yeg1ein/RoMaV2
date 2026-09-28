#!/bin/bash -l
# Fine-tune RoMa v2 on MINIMA-format data (MegaDepth + MegaDepth-Syn modalities).
# Counterpart of MINIMA/train_orders/minima_roma.sh, see minima_romav2_train_config.yaml.

set -e

ORDER_PATH=$(dirname $(readlink -f "$0"))
ROMA_DIR="${ORDER_PATH}/.."
CONFIG_PATH="${CONFIG_PATH:-${ORDER_PATH}/minima_romav2_train_config.yaml}"
SAVE_DIR="${SAVE_DIR:-${ROMA_DIR}/train_results/minima_romav2}"

export PYTHONPATH=${ROMA_DIR}/src:$PYTHONPATH
cd $ROMA_DIR
if [[ -d "${ROMA_DIR}/.venv/bin" ]]; then
    export PATH="${ROMA_DIR}/.venv/bin:$PATH"
fi

if ! command -v yq >/dev/null; then
    echo "[Error] 'yq' is required to read ${CONFIG_PATH} (github.com/mikefarah/yq or pip install yq)." >&2
    exit 1
fi

# Quotes are stripped so that both the Go and the jq-based yq can be used.
yq_str() { yq "$1" "$CONFIG_PATH" | sed "s/[\"']//g"; }

# Data
readarray -t modality_list < <(yq_str '.MINIMA_ROMAV2.dataset.modality[]')
modality_args=(--modality-list "${modality_list[@]}")
data_root=$(yq_str '.MINIMA_ROMAV2.dataset.data_root')
modality_root=$(yq_str '.MINIMA_ROMAV2.dataset.modality_root')
val=$(yq_str '.MINIMA_ROMAV2.dataset.val')

# DDP
n_nodes=$(yq_str '.MINIMA_ROMAV2.train.n_nodes')
n_gpus_per_node=$(yq_str '.MINIMA_ROMAV2.train.n_gpus_per_node')

# Optimisation
stage=$(yq_str '.MINIMA_ROMAV2.train.stage')
exp_name=$(yq_str '.MINIMA_ROMAV2.train.exp_name')
steps=$(yq_str '.MINIMA_ROMAV2.train.steps')
batch_size=$(yq_str '.MINIMA_ROMAV2.train.batch_size')
accum=$(yq_str '.MINIMA_ROMAV2.train.accum')
lr=$(yq_str '.MINIMA_ROMAV2.train.lr')
workers=$(yq_str '.MINIMA_ROMAV2.train.workers')
bidirectional=$(yq_str '.MINIMA_ROMAV2.train.bidirectional')
log_every=$(yq_str '.MINIMA_ROMAV2.train.log_every')
ckpt_every=$(yq_str '.MINIMA_ROMAV2.train.ckpt_every')
eval_every=$(yq_str '.MINIMA_ROMAV2.train.eval_every')
seed=$(yq_str '.MINIMA_ROMAV2.train.seed')
wandb=$(yq_str '.MINIMA_ROMAV2.train.wandb')

# Weights
pretrained=$(yq_str '.MINIMA_ROMAV2.weight.pre_trained')
init_path=$(yq_str '.MINIMA_ROMAV2.weight.init_path')
resume=$(yq_str '.MINIMA_ROMAV2.weight.resume_from_checkpoint')

extra_args=()

if [[ "$resume" == "true" && -n "$init_path" && "$init_path" != "null" ]]; then
    echo "[Error] Cannot enable both 'init_path' and 'resume_from_checkpoint'." >&2
    exit 1
fi

if [[ "$pretrained" != "true" ]]; then
    extra_args+=(--from-scratch)
fi
if [[ "$bidirectional" == "true" ]]; then
    extra_args+=(--bidirectional)
fi
if [[ "$wandb" == "true" ]]; then
    extra_args+=(--wandb)
fi
if [[ -n "$modality_root" && "$modality_root" != "null" ]]; then
    extra_args+=(--modality-root "$modality_root")
fi
if [[ -n "$val" && "$val" != "null" ]]; then
    extra_args+=(--val "$val")
fi

# Output workspace: reuse it when resuming, otherwise never overwrite an existing run.
OUT_DIR="${SAVE_DIR}/workspace_${exp_name}"
if [[ "$resume" == "true" ]]; then
    if [[ ! -f "${OUT_DIR}/checkpoint.pt" ]]; then
        echo "[Error] No checkpoint to resume at ${OUT_DIR}/checkpoint.pt." >&2
        exit 1
    fi
    extra_args+=(--resume "${OUT_DIR}/checkpoint.pt")
else
    suffix=2
    while [[ -d "$OUT_DIR" ]]; do
        OUT_DIR="${SAVE_DIR}/workspace_${exp_name}_${suffix}"
        suffix=$((suffix + 1))
    done
    if [[ -n "$init_path" && "$init_path" != "null" ]]; then
        extra_args+=(--init "$init_path")
    fi
fi
mkdir -p "$OUT_DIR"

torchrun --nproc_per_node=${n_gpus_per_node} --nnodes=${n_nodes} \
    -m romav2.train.minima \
    --stage ${stage} \
    --out ${OUT_DIR} \
    --megadepth ${data_root} \
    "${modality_args[@]}" \
    --steps ${steps} \
    --batch-size ${batch_size} \
    --accum ${accum} \
    --lr ${lr} \
    --workers ${workers} \
    --log-every ${log_every} \
    --ckpt-every ${ckpt_every} \
    --eval-every ${eval_every} \
    --seed ${seed} \
    "${extra_args[@]}"
