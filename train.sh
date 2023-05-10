#!/bin/bash

export MODEL_NAME="/home/ywz/data/models/stable-diffusion-v1-4"
export DATASET_NAME="/home/ywz/data/dota2/heroes/train"

accelerate launch --mixed_precision="fp16" models/train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=1234 \
  --output_dir="/home/ywz/data/dota2/tmp" \
  --validation_prompt="a dota 2 hero"
