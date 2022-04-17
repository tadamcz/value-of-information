#!/bin/bash
LOG_FILE=pytest_extra_slow.log
USE_MULTIPLE_SEEDS=1
exec > >(tee -a ${LOG_FILE}) 2>&1
echo "running $(basename $BASH_SOURCE) at $(date)"
pytest -m "extra_slow"