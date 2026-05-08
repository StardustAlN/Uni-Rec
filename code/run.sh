#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- OneTrans Training Configuration ----
python3 -u "${SCRIPT_DIR}/train_onetrans.py" \
    --train_data "../data_sample/demo_1000.parquet" \
    --val_data "../data_sample/demo_1000.parquet" \
    --schema_path "../data_sample/schema.json" \
    --output_dir "./outputs_onetrans" \
    --batch_size 32 \
    --epochs 3 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --num_workers 4 \
    --d_model 128 \
    --n_heads 4 \
    --d_ff 512 \
    --n_layers 4 \
    --max_seq_len 256 \
    "$@"
