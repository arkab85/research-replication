#!/bin/bash
# container: 2 cores. cw audit (2 parts of a 4-part split: parts 0,1), then apply cw FR/RF, then power parts 0,1 of 4
cd "$(dirname "$0")/.."
L=options/logs
python3 options/options_size.py cw 40 0 4 > $L/size_cw_0.log 2>&1 & python3 options/options_size.py cw 40 1 4 > $L/size_cw_1.log 2>&1; wait
echo "cw audit parts 0,1 done $(date -u)" >> $L/progress.log
python3 options/options_apply.py cw FR > $L/apply_cw_FR.log 2>&1 & python3 options/options_apply.py cw RF > $L/apply_cw_RF.log 2>&1; wait
echo "cw apply done $(date -u)" >> $L/progress.log
python3 options/options_power.py 20 0 4 > $L/power_0.log 2>&1 & python3 options/options_power.py 20 1 4 > $L/power_1.log 2>&1; wait
echo "power parts 0,1 done $(date -u)" >> $L/progress.log
