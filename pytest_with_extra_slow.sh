#!/bin/bash
CURRENTDATE=`date +"%Y-%m-%d_%T"`
mkdir logs
LOG_FILE=logs/${CURRENTDATE}_$(basename $BASH_SOURCE .sh).log
echo "log file: $LOG_FILE"
exec > >(tee -a ${LOG_FILE}) 2>&1
export N_RAND_SEED=10
echo "running $(basename $BASH_SOURCE) at $CURRENTDATE with revision $(git rev-parse HEAD)"
pytest -n auto
pytest -m "extra_slow and not extra_mem" -n auto
# As of 8da040ef, each of the extra_mem tests can take about 5 gb of memory
pytest -m "extra_slow and extra_mem" -n 10