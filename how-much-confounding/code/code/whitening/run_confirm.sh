#!/bin/sh
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python confirm_study.py W 2 >> results/confirm_log.txt 2>&1
python confirm_study.py R 2 >> results/confirm_log.txt 2>&1
python confirm_study.py J 2 >> results/confirm_log.txt 2>&1
echo ALLDONE >> results/confirm_log.txt
