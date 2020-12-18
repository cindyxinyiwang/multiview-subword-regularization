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

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --mem=15GB

REPO=$PWD
GPU=${1:-0}
#MODEL=${2:-bert-base-multilingual-cased}
MODEL=${2:-xlm-roberta-base}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}

#export CUDA_VISIBLE_DEVICES=$GPU
TASK='panx'
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
TRAIN_LANGS="en"
NUM_EPOCHS=10
MAX_LENGTH=128
MLM_WEIGHT=0
MLM_LANG='zh,ko,ja'
OPTIM='Adam'
MLM_START=5
MLM_END=10
LR=2e-5
ATTN_T=1
UPDATE_PRETRAIN=0

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

DATA_DIR=$DATA_DIR/${TASK}/${TASK}_processed_maxlen${MAX_LENGTH}/

for SEED in 1 2 3 4 5;
do
if [ $MLM_WEIGHT == 0 ]; then
  OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}_optim${OPTIM}_up${UPDATE_PRETRAIN}_s${SEED}/"
else
  OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}-LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-TrainLang${TRAIN_LANGS}_MLMW${MLM_WEIGHT}_MLML${MLM_LANG}_MLMS${MLM_START}_MLME${MLM_END}_optim${OPTIM}_s${SEED}/"
fi

mkdir -p $OUTPUT_DIR
python $REPO/third_party/run_tag.py \
  --do_train \
  --do_eval \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --seed $SEED \
  --learning_rate $LR \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --optimizer $OPTIM \
  --mlm_weight $MLM_WEIGHT \
  --mlm_lang $MLM_LANG \
  --mlm_start_epoch $MLM_START \
  --mlm_end_epoch $MLM_END \
  --attention_t $ATTN_T \
  --update_pretrained_epoch $UPDATE_PRETRAIN \
  --save_only_best_checkpoint $LC
done
