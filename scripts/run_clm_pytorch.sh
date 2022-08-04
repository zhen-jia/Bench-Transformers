#!/bin/bash
# Origninal from Zhengyuan Shen's repo: https://gitlab.aws.dev/donshen/dist-training-benchmark/-/tree/master/
# Model Specs

# MODEL_TYPE='gpt2'
# TOKENIZER_NAME='gpt2'

MODEL_TYPE='gpt_neo'
TOKENIZER_NAME='EleutherAI/gpt-neo-1.3B'
NUM_LAYERS=24
HIDDEN_SIZE=2048

# Data Specs
DATASET_NAME='wikitext'
DATASET_CONFIG_NAME="wikitext-103-raw-v1"

# Training Specs
NUM_TRAIN_EPOCHS=2
LEARNING_RATE=5e-6
#PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_TRAIN_BATCH_SIZE=4
BLOCK_SIZE=512
GRADIENT_CHECKPOINTING=True
FP16=True

# Paths
SCRIPT_PATH="pytorch/run_clm.py"
ZERO_CONFIG_PATH="config/ds_config_zero2.json"
OUTPUT_PATH="./output_dir/clm-pytorch-${MODEL_TYPE}-l${NUM_LAYERS}-h${HIDDEN_SIZE}-s${BLOCK_SIZE}-b${PER_DEVICE_TRAIN_BATCH_SIZE}"

# Command
deepspeed ${SCRIPT_PATH} \
--deepspeed ${ZERO_CONFIG_PATH} \
--output_dir ${OUTPUT_PATH} \
--overwrite_output_dir \
--model_type ${MODEL_TYPE} --tokenizer_name ${TOKENIZER_NAME} \
--num_layers ${NUM_LAYERS} --hidden_size ${HIDDEN_SIZE} \
--do_train \
--dataset_name ${DATASET_NAME} \
--dataset_config_name ${DATASET_CONFIG_NAME} \
--num_train_epochs ${NUM_TRAIN_EPOCHS} --learning_rate ${LEARNING_RATE} \
--per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
--block_size ${BLOCK_SIZE} \
--fp16 ${FP16} --gradient_checkpointing ${GRADIENT_CHECKPOINTING}  

