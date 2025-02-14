#!/usr/bin/env bash

set -x
set -e

TASK="FB15k237"
EXPERIMENT="DIST-True"


DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}/${TASK}_${EXPERIMENT}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

nohup python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 1e-5 \
--use-link-graph \
--train-path "$DATA_DIR/train.txt.json" \
--valid-path "$DATA_DIR/valid.txt.json" \
--task ${TASK} \
--batch-size 512 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--model-plus \
--use-self-negative \
--pre-batch 2 \
--epochs 20 \
--workers 4 \
--max-to-keep 2 "$@" > "${OUTPUT_DIR}_train.log" 2>&1 &

# --max-to-keep 5 "$@"
# --finetune-t 

