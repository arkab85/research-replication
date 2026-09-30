#!/bin/bash
# HOLDOUT_PROTOCOL.md Section 7: audits first, then the real outcomes; two parts side by side.
cd "$(dirname "$0")/.."
L=flows/logs
run2 () { python3 flows/holdout_size.py $1 $2 $3 120 0 2 > $L/size_$1_$2_$3_0.log 2>&1 & python3 flows/holdout_size.py $1 $2 $3 120 1 2 > $L/size_$1_$2_$3_1.log 2>&1; wait; }
run2 A oi_growth ret; run2 A oi_growth vol; run2 A nc_short_chg vol
echo "A audits done $(date -u)" >> $L/progress.log
python3 flows/holdout_apply.py A oi_growth r > $L/apply_A_oi_r.log 2>&1 & python3 flows/holdout_apply.py A oi_growth rv > $L/apply_A_oi_rv.log 2>&1; wait
python3 flows/holdout_apply.py A nc_short_chg rv > $L/apply_A_short_rv.log 2>&1
echo "A apply done $(date -u)" >> $L/progress.log
run2 B oi_growth ret; run2 B oi_growth vol; run2 B nc_short_chg vol
echo "B audits done $(date -u)" >> $L/progress.log
python3 flows/holdout_apply.py B oi_growth r > $L/apply_B_oi_r.log 2>&1 & python3 flows/holdout_apply.py B oi_growth rv > $L/apply_B_oi_rv.log 2>&1; wait
python3 flows/holdout_apply.py B nc_short_chg rv > $L/apply_B_short_rv.log 2>&1
echo "B apply done $(date -u)" >> $L/progress.log
