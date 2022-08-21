#!/bin/bash

cd $(dirname "$0")

source ../../conf/variables.sh

bash start_extraction.sh

bash get_labels.sh

bash train_valid_split.sh

exit 0