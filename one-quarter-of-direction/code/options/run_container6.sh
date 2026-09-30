#!/bin/bash
cd "$(dirname "$0")/.."
L=options/logs
python3 options/options_size.py retvol 20 0 2 > $L/size_retvol_0.log 2>&1 & python3 options/options_size.py retvol 20 1 2 > $L/size_retvol_1.log 2>&1; wait
echo "retvol audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py retvol FR > $L/apply_retvol_FR.log 2>&1 & python3 options/options_apply.py retvol RF > $L/apply_retvol_RF.log 2>&1; wait
echo "retvol apply done $(date -u)" >> $L/progress.log
