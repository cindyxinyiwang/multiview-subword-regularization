#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script to train a model on SQuAD v1.1 or the English TyDiQA-GoldP train data.

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=50GB

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
SRC=${2:-squad}
TGT=${3:-xquad}
GPU=${4:-0}
DATA_DIR=${5:-"$REPO/download/"}
OUT_DIR=${6:-"$REPO/outputs/"}

MAXL=384
LR=3e-5
NUM_EPOCHS=2
BPE_DROP=0.2
KL=0.2 
KL_T=1
KL_TB=0
KL_TG=0
KL_SG=0


if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlm-roberta"
fi

# Model path where trained model should be stored
for SEED in 2 3 4 5;
do
  MODEL_PATH=$OUT_DIR/$SRC/${MODEL}_LR${LR}_EPOCH${NUM_EPOCHS}_maxlen${MAXL}_mbped${BPE_DROP}_kl${KL}_klt${KL_T}_kltb${KL_TB}_kltg${KL_TG}_klsg${KL_SG}_s${SEED}
  mkdir -p $MODEL_PATH
  # Train either on the SQuAD or TyDiQa-GoldP English train file
  if [ $SRC == 'squad' ]; then
    TRAIN_FILE=${DATA_DIR}/squad/train-v1.1.json
    PREDICT_FILE=${DATA_DIR}/squad/dev-v1.1.json
  else
    TRAIN_FILE=${DATA_DIR}/tydiqa/tydiqa-goldp-v1.1-train/tydiqa.goldp.en.train.json
    PREDICT_FILE=${DATA_DIR}/tydiqa/tydiqa-goldp-v1.1-dev/tydiqa.en.dev.json
  fi
  
  # train
  #CUDA_VISIBLE_DEVICES=$GPU python third_party/run_squad.py \
  python third_party/run_mv_squad.py \
    --data_dir  ${DATA_DIR}/${SRC} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL} \
    --do_lower_case \
    --do_train \
    --do_eval \
    --train_file ${TRAIN_FILE} \
    --predict_file ${PREDICT_FILE} \
    --per_gpu_train_batch_size 4 \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --max_seq_length $MAXL \
    --doc_stride 128 \
    --save_steps -1 \
    --overwrite_output_dir \
    --gradient_accumulation_steps 4 \
    --warmup_steps 500 \
    --output_dir ${MODEL_PATH} \
    --weight_decay 0.0001 \
    --threads 1 \
    --log_file ${MODEL_PATH}/train.log \
    --seed $SEED \
    --train_lang en \
    --bpe_dropout $BPE_DROP \
    --kl_weight $KL \
    --kl_t $KL_T \
    --kl_t_scale_both $KL_TB \
    --kl_t_scale_grad $KL_TG \
    --kl_stop_grad $KL_SG \
    --eval_lang en
  
  
  # predict
  #bash scripts/predict_qa.sh $MODEL $MODEL_TYPE $MODEL_PATH $TGT $GPU $DATA_DIR
  bash scripts/predict_qa.sh $MODEL $MODEL_TYPE $MODEL_PATH xquad $GPU $DATA_DIR
  bash scripts/predict_qa.sh $MODEL $MODEL_TYPE $MODEL_PATH mlqa $GPU $DATA_DIR
done
