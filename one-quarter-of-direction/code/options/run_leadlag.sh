#!/bin/bash
cd "$(dirname "$0")/.."
L=options/logs
while ! grep -q "cw apply done" $L/progress.log 2>/dev/null; do sleep 30; done
python3 options/options_size.py big 40 0 1 > $L/size_big.log 2>&1
echo "leadlag audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py big FR > $L/apply_big_FR.log 2>&1; python3 options/options_apply.py big RF > $L/apply_big_RF.log 2>&1
echo "leadlag apply done $(date -u)" >> $L/progress.log
python3 options/options_power_ll.py 20 0 1 > $L/power_ll.log 2>&1
echo "leadlag power done $(date -u)" >> $L/progress.log
