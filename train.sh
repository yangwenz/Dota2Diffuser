#!/bin/bash

export MODEL_NAME="/home/ywz/data/models/stable-diffusion-v1-5"
export DATASET_DIR="/home/ywz/data/dota2/heroes/selected/train"
export MODEL_DIR="/home/ywz/data/dota2/model_2"
export INIT_MODEL_DIR=$MODEL_DIR

# training_batch_size: 1 or 2
# num_train_epochs: 100, 120, or 150

accelerate launch --mixed_precision="fp16" train_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
  --caption_column="text" \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --lora_rank=4 \
  --checkpointing_steps=5000 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=1234 \
  --output_dir=$MODEL_DIR \
  --init_model_dir=$INIT_MODEL_DIR \
  --validation_prompt="lina_dota, full body, best quality, ultra detailed, masterpiece" \
  --validation_label_index=98 \
  --num_validation_images=4 \
  --validation_epochs=10
