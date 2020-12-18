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

# Script to obtain predictions using a trained model on XQuAD, TyDi QA, and MLQA.
REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_TYPE=${2:-bert}
MODEL_PATH=${3}
TGT=${4:-xquad}
GPU=${5:-0}
DATA_DIR=${6:-"$REPO/download/"}

if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model path ${MODEL_PATH} does not exist."
  exit
fi

DIR=${DATA_DIR}/${TGT}/
PREDICTIONS_DIR=${MODEL_PATH}/predictions
PRED_DIR=${PREDICTIONS_DIR}/$TGT/
mkdir -p "${PRED_DIR}"

if [ $TGT == 'xquad' ]; then
  langs=( en es de el ru tr ar vi th zh hi )
  #langs=( zh )
elif [ $TGT == 'mlqa' ]; then
  langs=( en es de ar hi vi zh )
  #langs=( zh )
elif [ $TGT == 'tydiqa' ]; then
  langs=( en ar bn fi id ko ru sw te )
fi

echo "************************"
echo ${MODEL}
echo "************************"

echo
echo "Predictions on $TGT"
for lang in ${langs[@]}; do
  echo "  $lang "
  if [ $TGT == 'xquad' ]; then
    TEST_FILE=${DIR}/xquad.$lang.json
  elif [ $TGT == 'mlqa' ]; then
    TEST_FILE=${DIR}/MLQA_V1/test/test-context-$lang-question-$lang.json
  elif [ $TGT == 'tydiqa' ]; then
    TEST_FILE=${DIR}/tydiqa-goldp-v1.1-dev/tydiqa.$lang.dev.json
  fi

  #CUDA_VISIBLE_DEVICES=${GPU} python third_party/run_squad.py \
  #  --do_lower_case \
  python third_party/run_squad.py \
    --data_dir $DIR \
    --model_name ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --do_test \
    --eval_lang ${lang} \
    --predict_file "${TEST_FILE}" \
    --log_file ${MODEL_PATH}/train.log \
    --do_final 0 \
    --output_dir "${PRED_DIR}"
  python third_party/run_squad.py \
    --data_dir $DIR \
    --model_name ${MODEL} \
    --model_type ${MODEL_TYPE} \
    --model_name_or_path ${MODEL_PATH} \
    --do_test \
    --eval_lang ${lang} \
    --predict_file "${TEST_FILE}" \
    --log_file ${MODEL_PATH}/train.log \
    --do_final 1 \
    --output_dir "${PRED_DIR}"

    #--output_dir "${PRED_DIR}" &> /dev/null
done

