#!/usr/bin/env bash

set -x
set -e

TASK="WN18RR"
# EXPERIMENT="Baseline_Latter"
EXPERIMENT="Classifier_Top20_CN100_EP50_Alpha2_LR5-5_MergeTuckerRRCondidateBTT20_FintueT_NoSelfSeg"
# EXPERIMENT="Classifier_Top0_CN100_EP50_Alpha0.5_LR5-5_LogitsCondidateTBALL_FintueT_NoSelfSeg_TuckerNoTrpBnNoDP"
# EXPERIMENT="TEST"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_${EXPERIMENT}_$(date +%F-%H%M.%S)"
fi

[ ! -d ${DIR} ] && mkdir ${DIR}

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

nohup python3 -u main.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 5e-5 \
--use-link-graph \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--task ${TASK} \
--batch-size 256 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--pre-batch 0 \
--epochs 50 \
--workers 4 \
--model-plus \
--topk 50 \
--alpha 2 \
--finetune-t \
--class-num 100 \
--candidate-num 20 \
--max-to-keep 3 "$@" > "${OUTPUT_DIR}_train.log" 2>&1 &

# --use-self-negative \
# --possibility \


