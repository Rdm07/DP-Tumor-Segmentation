# !/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

out_folders="heatmap_jsons heatmap_txt json log patch-level-color patch-level-lym patch-level-merged patch-level-nec"
for i in ${out_folders}; do
	if [ ! -d ${PRED_DATA_DIR}/$i ]; then
		mkdir -p ${PRED_DATA_DIR}/$i
	fi
done

if [ ! -d ${PRED_DATA_DIR}/patches ]; then
	mkdir -p ${PRED_DATA_DIR}/patches;
fi
wait;

cd ..

cd patch_extraction_cancer_40X
nohup bash start.sh 
cd ..

cd prediction
nohup bash start.sh
cd ..

wait;

cd heatmap_gen
nohup bash start.sh
cd ..

wait;

exit 0
