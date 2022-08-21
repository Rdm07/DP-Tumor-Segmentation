#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

FN=$1
LYM_FOLDER=${PRED_DATA_DIR}/patch-level-lym/
OUT_FOLDER=${PRED_DATA_DIR}/patch-level-merged/

awk '{
    print $1, $2, $3, 0.0;
}' ${LYM_FOLDER}/${FN} > ${OUT_FOLDER}/${FN} 2>&1

exit 0

