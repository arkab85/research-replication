import sys, re, os
sys.argv = ['x']
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '110_buyout.py')).read().split('if __name__')[0])

for fn in ['1033012_0001033012-22-000025_fbc-20211231.htm']:
    t = flat(os.path.join(C, fn))
    print(fn)
    for m in re.finditer(r'Loans with government guarantees', t):
        seg = re.sub(r'\s+', ' ', t[m.start():m.start() + 150])
        if re.match(r'Loans with government guarantees\s*(repurchase options)?\s*\|\s*[\$\d]', seg):
            print('  >', re.sub(r'\s+', ' ', t[max(0, m.start() - 150):m.start() + 210]))
            print('  ---')
