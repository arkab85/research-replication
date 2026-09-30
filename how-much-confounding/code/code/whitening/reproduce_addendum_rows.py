"""Re-run the first stored replication of one design in each addendum part from its seed."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from confirm_addendum import run_RB, run_JW, run_WB, RES
out = {}
for part, name, f in (('RB', 'R4_logsv_n600', run_RB), ('JW', 'J5_chi40_n300', run_JW), ('WB', 'V6L_logsv_n600', run_WB)):
    stored = json.loads(open(os.path.join(RES, f'addendum_{part}_{name}.jsonl')).readline())
    again = json.loads(json.dumps(f((name, stored['seed']))))
    assert again == stored, (part, name)
    out[f'{part}:{name}'] = 'reproduced'
json.dump(out, open(os.path.join(RES, 'reproduction_check_addendum.json'), 'w'), indent=2); print(out)
