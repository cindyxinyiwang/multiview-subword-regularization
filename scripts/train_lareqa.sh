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

python third_party/run_retrieval_qa.py \
  --model_type bert-retrieval \
  --model_name_or_path bert-base-multilingual-cased \
  --do_lower_case \
  --do_train \
  --do_eval \
  --train_file download/squad/train-v1.1.json \
  --predict_file download/squad/dev-v1.1.json \
  --per_gpu_train_batch_size 4 \
  --learning_rate 0.001 \
  --num_train_epochs 1 \
  --max_seq_length 128 \
  --save_steps 2 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 4 \
  --warmup_steps 500 \
  --output_dir runs/lareqa-train \
  --weight_decay 0.0001 \
  --threads 8 \
  --train_lang en \
  --eval_lang en
