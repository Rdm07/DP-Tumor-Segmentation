# !/bin/bash

source ../conf/variables.sh

i=20

for folder in ${SVS_QUEUE_FOLDER}/*
do
    echo ${folder}
    echo ${i}
    sudo mv ${folder} ${DOWNLOAD_FOLDER}
    ((i--))
    if [ ${i} == 0 ]; then
        break
    fi
done

sudo mv ${DOWNLOAD_FOLDER}/*/*.svs ${TRAINING_SVS}

exit 0
