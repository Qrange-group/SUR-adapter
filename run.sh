export CUDA=$1
export LLM="13B"
export LLM_LAYER=39
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INFO="test"
export TRAIN_DIR="SURD"
export OUTPUT_DIR="fp16"
export SAVE_STEP=100
export BATCH_SIZE=1

CUDA_VISIBLE_DEVICES=$CUDA accelerate launch SUR_adapter_train.py \
  --mixed_precision="fp16" \
  --info=$INFO \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --llm=$LLM \
  --llm_layer=$LLM_LAYER \
  --checkpointing_steps=$SAVE_STEP \
  --train_batch_size=$BATCH_SIZE \
  --resolution=512 --center_crop --random_flip \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=5000 \
  --learning_rate=1e-05 \
  --prompt_weight=1e-05 \
  --llm_weight=1e-05 \
  --adapter_weight=1e-01 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 

