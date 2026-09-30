"""Re-run the first stored replication of one design in each confirmatory part and
check it is reproduced exactly from its seed."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from confirm_study import run_W, run_J, run_R, RES
out = {}
for part, name, f in (('W', 'V6_logsv_n600', run_W), ('J', 'J3_mk95_d4_n300', run_J), ('R', 'R1_iid_d0_n600', run_R)):
    stored = json.loads(open(os.path.join(RES, f'confirm_{part}_{name}.jsonl')).readline())
    again = json.loads(json.dumps(f((name, stored['seed']))))
    assert again == stored, (part, name)
    out[f'{part}:{name}'] = 'reproduced'
json.dump(out, open(os.path.join(RES, 'reproduction_check.json'), 'w'), indent=2); print(out)
