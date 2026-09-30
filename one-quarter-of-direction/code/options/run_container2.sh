#!/bin/bash
cd "$(dirname "$0")/.."
L=options/logs
# wait for the two running cw audit parts (0,1 of 4) to finish
while pgrep -f "options_size.py cw 40 [01] 4" > /dev/null; do sleep 20; done
echo "cw audit parts 0,1 done $(date -u)" >> $L/progress.log
python3 options/options_size.py cw 40 2 4 > $L/size_cw_2.log 2>&1 & python3 options/options_size.py cw 40 3 4 > $L/size_cw_3.log 2>&1; wait
echo "cw audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py cw FR > $L/apply_cw_FR.log 2>&1 & python3 options/options_apply.py cw RF > $L/apply_cw_RF.log 2>&1; wait
echo "cw apply done $(date -u)" >> $L/progress.log
python3 options/options_size.py skew 40 0 2 > $L/size_skew_0.log 2>&1 & python3 options/options_size.py skew 40 1 2 > $L/size_skew_1.log 2>&1; wait
echo "skew audit done $(date -u)" >> $L/progress.log
python3 options/options_apply.py skew FR > $L/apply_skew_FR.log 2>&1 & python3 options/options_apply.py skew RF > $L/apply_skew_RF.log 2>&1; wait
echo "skew apply done $(date -u)" >> $L/progress.log
python3 options/options_power.py 20 0 2 > $L/power_0.log 2>&1 & python3 options/options_power.py 20 1 2 > $L/power_1.log 2>&1; wait
echo "power done $(date -u)" >> $L/progress.log
