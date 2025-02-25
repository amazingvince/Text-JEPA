#!/bin/bash
# Script to train Text-JEPA with memory optimization features

# Set experiment name
EXPERIMENT_NAME="jepa_large_model_memory_optimized"

# Memory optimization settings
# Accumulation steps of 4 with batch size 8 = effective batch size 32
GRAD_ACCUM_STEPS=4
# Enable gradient checkpointing
ENABLE_CHECKPOINTING=true

# Create output directories
mkdir -p outputs/$EXPERIMENT_NAME
mkdir -p logs/$EXPERIMENT_NAME

# Run training with memory optimizations
python train.py \
  --config config/default.yaml \
  --output_dir outputs/$EXPERIMENT_NAME \
  --log_dir logs/$EXPERIMENT_NAME \
  --subset en \
  --streaming \
  --experiment_name $EXPERIMENT_NAME \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --use_wandb \
  $([ "$ENABLE_CHECKPOINTING" = true ] && echo "--gradient_checkpointing")

# Alternatively, you can configure these settings in the config file
# and run without command-line arguments:
# python train.py --config config/memory_optimized.yaml

# Note: You may need to adjust the learning rate for larger effective batch sizes