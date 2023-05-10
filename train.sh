#!/bin/bash

export MODEL_NAME="/home/ywz/data/models/stable-diffusion-v1-4"
export DATASET_DIR="/home/ywz/data/dota2/heroes/train"
export MODEL_DIR="/home/ywz/data/dota2/tmp"

accelerate launch --mixed_precision="fp16" models/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --caption_column="text" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=2 \
  --num_train_epochs=100 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=0 \
  --output_dir=$MODEL_DIR \
  --validation_prompt="a dota 2 hero Crystal Maiden" \
  --num_validation_images=2 \
  --validation_epochs=10
