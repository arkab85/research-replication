#!/bin/sh
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python app_block_sensitivity.py >> results/addendum_log.txt 2>&1
python confirm_addendum.py RB 2 >> results/addendum_log.txt 2>&1
python confirm_addendum.py JW 2 >> results/addendum_log.txt 2>&1
python confirm_addendum.py WB 2 >> results/addendum_log.txt 2>&1
echo ALLDONE >> results/addendum_log.txt
