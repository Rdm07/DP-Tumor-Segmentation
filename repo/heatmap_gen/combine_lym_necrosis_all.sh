#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

LYM_FOLDER=${PRED_DATA_DIR}/patch-level-lym
NEC_FOLDER=${PRED_DATA_DIR}/patch-level-nec

for files in ${LYM_FOLDER}/*; do
    if [ ! -f ${files} ]; then continue; fi

    fn=`echo ${files} | awk -F'/' '{print $NF}'`
    if [ -f ${NEC_FOLDER}/${fn} ]; then
        bash combine_lym_necrosis.sh ${fn} > ${LOG_OUTPUT_FOLDER}/log.combine_lym_necrosis.txt 2>&1
    else
        bash combine_lym_no_necrosis.sh ${fn} > ${LOG_OUTPUT_FOLDER}/log.combine_lym_no_necrosis.txt 2>&1
    fi
done

exit 0
