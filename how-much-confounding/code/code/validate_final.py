"""Final numerical/source consistency checks without repeating 10,000 simulations."""
from pathlib import Path
import json,sys,numpy as np,ast
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'source'))
from feasible_inference import derivative_geometry
from jointproc import direction_parts,dictionary
rng=np.random.default_rng(532);e=rng.normal(size=60);z=rng.normal(size=60)
g,b=derivative_geometry(e,z,degree=2);p=direction_parts(e,z,dictionary(z))
assert np.allclose(g,p['GJ'],atol=1e-12) and np.allclose(b,p['B'],atol=1e-12)
o=json.loads((P/'code/oil_final.json').read_text());assert o['A']['m']==384 and o['A']['n']==213 and o['B']['m']==288 and o['B']['n']==309
expected=[(2.99,2.57,-.42,-34.6,33.1),(2.14,2,-.15,-31.5,30.9),(1.62,2.34,.72,-120.2,126),(1.70,2.21,.52,-120.9,125)]
actual=o['A']['rows']+o['B']['rows']
for r,t in zip(actual,expected):
 assert np.allclose([r['Hf']*1000,r['Hb']*1000,r['D']*1000],t[:3],atol=.0051)
 assert np.allclose([r['L']*1000,r['U']*1000],t[3:],atol=.051)
 assert r['L']<0 and r['Lfix']<0 and r['L3']<0
for folder in ['source','code']:
 for f in (P/folder).glob('*.py'):ast.parse(f.read_text())
report={'joint_derivative_implementation_matches_previous_validated_code':True,'oil_table_rounding_verified':True,'oil_splits_verified':True,'oil_lower_certificates_zero':True,'source_syntax_valid':True}
(P/'audit/final_numerical_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
