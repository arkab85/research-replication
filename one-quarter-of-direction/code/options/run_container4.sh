#!/bin/bash
cd "$(dirname "$0")/.."
L=options/logs
while ps -eo cmd | grep -v grep | grep -q "options_size.py ret 40"; do sleep 20; done
echo "ret-IV audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py ret FR > $L/apply_ret_FR.log 2>&1 & python3 options/options_apply.py ret RF > $L/apply_ret_RF.log 2>&1; wait
echo "ret-IV apply done $(date -u)" >> $L/progress.log
python3 options/options_size.py spy 40 0 2 > $L/size_spy_0.log 2>&1 & python3 options/options_size.py spy 40 1 2 > $L/size_spy_1.log 2>&1; wait
echo "spy audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py spy FR > $L/apply_spy_FR.log 2>&1 & python3 options/options_apply.py spy RF > $L/apply_spy_RF.log 2>&1; wait
echo "spy apply done $(date -u)" >> $L/progress.log
python3 options/options_size.py retvol 40 0 2 > $L/size_retvol_0.log 2>&1 & python3 options/options_size.py retvol 40 1 2 > $L/size_retvol_1.log 2>&1; wait
echo "retvol audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py retvol FR > $L/apply_retvol_FR.log 2>&1 & python3 options/options_apply.py retvol RF > $L/apply_retvol_RF.log 2>&1; wait
echo "retvol apply done $(date -u)" >> $L/progress.log
python3 options/options_size.py skew 40 0 2 > $L/size_skew_0.log 2>&1 & python3 options/options_size.py skew 40 1 2 > $L/size_skew_1.log 2>&1; wait
echo "skew audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py skew FR > $L/apply_skew_FR.log 2>&1 & python3 options/options_apply.py skew RF > $L/apply_skew_RF.log 2>&1; wait
echo "skew apply done $(date -u)" >> $L/progress.log
