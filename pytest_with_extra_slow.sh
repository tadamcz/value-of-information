#!/bin/bash
CURRENTDATE=`date +"%Y-%m-%d_%T"`
mkdir logs
LOG_FILE=logs/${CURRENTDATE}_$(basename $BASH_SOURCE .sh).log
echo "log file: $LOG_FILE"
exec > >(tee -a ${LOG_FILE}) 2>&1
export N_RAND_SEED=10
echo "running $(basename $BASH_SOURCE) at $CURRENTDATE with revision $(git rev-parse HEAD)"
pytest -v -n auto
pytest -v -m "extra_slow" -n auto