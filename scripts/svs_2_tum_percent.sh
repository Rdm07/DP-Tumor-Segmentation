#!/bin/bash

source ../conf/variables.sh

for folder in ${PRED_DATA_DIR}/patches/*
do
	((f=f+1))
	if [ -e ${folder}/extraction_done.txt ]; then
		((e=e+1))
	else
		((i=i+1))
	fi
done

if [[ ${f} -eq ${e} ]]; then
    echo "Continuing without Extraction"
	bash ${BASE_DIR}/scripts/svs_2_heatmap_wo_extraction.sh
else
    echo "Continuing with Extraction"
	bash ${BASE_DIR}/scripts/svs_2_heatmap.sh
fi

cd get_grayscale_heatmaps

bash start.sh

cd ..

bash ${BASE_DIR}/scripts/get_tum_percent.sh

exit 0
