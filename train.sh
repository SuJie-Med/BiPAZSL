#!/bin/bash
source activate apex

python -u train.py --config-file config/sun.yaml

#bash train.sh > aaa_sun.log 2>&1
