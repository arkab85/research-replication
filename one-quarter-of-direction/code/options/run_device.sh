#!/bin/bash
# device VM: 2 cores. cw audit parts 2,3 of 4; skew audit all 4 parts (2 at a time); apply skew FR/RF; power parts 2,3
cd "$(dirname "$0")/.."
L=options/logs; mkdir -p $L
python3 options/options_size.py cw 40 2 4 > $L/size_cw_2.log 2>&1 & python3 options/options_size.py cw 40 3 4 > $L/size_cw_3.log 2>&1; wait
echo "cw audit parts 2,3 done $(date -u)" >> $L/progress_device.log
python3 options/options_size.py skew 40 0 2 > $L/size_skew_0.log 2>&1 & python3 options/options_size.py skew 40 1 2 > $L/size_skew_1.log 2>&1; wait
echo "skew audit done $(date -u)" >> $L/progress_device.log
python3 options/options_apply.py skew FR > $L/apply_skew_FR.log 2>&1 & python3 options/options_apply.py skew RF > $L/apply_skew_RF.log 2>&1; wait
echo "skew apply done $(date -u)" >> $L/progress_device.log
python3 options/options_power.py 20 2 4 > $L/power_2.log 2>&1 & python3 options/options_power.py 20 3 4 > $L/power_3.log 2>&1; wait
echo "power parts 2,3 done $(date -u)" >> $L/progress_device.log
