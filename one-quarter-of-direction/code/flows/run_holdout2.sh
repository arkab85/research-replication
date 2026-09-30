#!/bin/bash
cd "$(dirname "$0")/.."
L=flows/logs
while ! grep -q "B apply done" $L/progress.log 2>/dev/null; do sleep 30; done
python3 flows/holdout_size.py A r vol 120 0 2 > $L/size_A_r_vol_0.log 2>&1 & python3 flows/holdout_size.py A r vol 120 1 2 > $L/size_A_r_vol_1.log 2>&1; wait
echo "A r audit done $(date -u)" >> $L/progress.log
python3 flows/holdout_apply.py A r rv > $L/apply_A_r_rv.log 2>&1
echo "A r apply done $(date -u)" >> $L/progress.log
python3 flows/holdout_power.py A oi_growth 60 0 2 > $L/power_A_0.log 2>&1 & python3 flows/holdout_power.py A oi_growth 60 1 2 > $L/power_A_1.log 2>&1; wait
echo "A power done $(date -u)" >> $L/progress.log
