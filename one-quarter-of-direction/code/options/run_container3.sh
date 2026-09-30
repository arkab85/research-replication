#!/bin/bash
cd "$(dirname "$0")/.."
L=options/logs
python3 options/options_size.py ret 40 0 2 > $L/size_ret_0.log 2>&1 & python3 options/options_size.py ret 40 1 2 > $L/size_ret_1.log 2>&1; wait
echo "ret-IV audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py ret FR > $L/apply_ret_FR.log 2>&1 & python3 options/options_apply.py ret RF > $L/apply_ret_RF.log 2>&1; wait
echo "ret-IV apply done $(date -u)" >> $L/progress.log
python3 options/options_size.py skew 40 0 2 > $L/size_skew_0.log 2>&1 & python3 options/options_size.py skew 40 1 2 > $L/size_skew_1.log 2>&1; wait
echo "skew audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py skew FR > $L/apply_skew_FR.log 2>&1 & python3 options/options_apply.py skew RF > $L/apply_skew_RF.log 2>&1; wait
echo "skew apply done $(date -u)" >> $L/progress.log
