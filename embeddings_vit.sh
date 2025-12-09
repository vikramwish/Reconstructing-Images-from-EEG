#!/bin/bash

CHECKPOINT="/raid/datasets/tanaya/fm/phase3_22/best_val_accuracy_model_epoch_4.pt"
INPUT_DIR="/tmp/tanaya/alljoined"
OUTPUT_DIR="/raid/datasets/tanaya/fm/phase3_22/learned_embeddings"

# Get all subjects
SUBJECTS=($(ls -d $INPUT_DIR/sub-*))
GPUS=(1 2 3 )  # Use GPUs 2 and 3
N_GPUS=${#GPUS[@]}

echo "Processing ${#SUBJECTS[@]} subjects across $N_GPUS GPUs..."

# Process subjects in parallel
for ((i=0; i<${#SUBJECTS[@]}; i++)); do
    GPU_ID=${GPUS[$((i % N_GPUS))]}
    SUBJECT=$(basename ${SUBJECTS[$i]})

    echo "Processing $SUBJECT on GPU $GPU_ID"

    # Extract this subject on specific GPU
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 /raid/datasets/tanaya/fm/script_f/eeg_vit_embeddings.py \
        --checkpoint $CHECKPOINT \
        --input_shards $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --feature_type combined &

    # Limit concurrent processes to number of GPUs
    if (( (i+1) % N_GPUS == 0 )); then
        wait  # Wait for batch to complete
    fi
done

wait  # Wait for all
echo "All extractions complete!"