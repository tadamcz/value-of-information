#!/bin/bash
LOG_FILE=pytest_extra_slow.log
exec > >(tee -a ${LOG_FILE}) 2>&1
echo "running $(basename $BASH_SOURCE) at $(date)"
pytest -v -m "extra_slow" -n auto