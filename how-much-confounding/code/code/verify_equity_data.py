"""Spot-check the S&P 500 and VIX daily series used in Section 9 against published closing values.
Run after placing sp500.csv.gz (arch package data set) and vix-daily.csv (datasets/finance-vix) next to this file,
or point the paths below at series downloaded directly from S&P Dow Jones Indices and Cboe."""
import pandas as pd
sp=pd.read_csv('sp500.csv.gz',parse_dates=[0],index_col=0); vx=pd.read_csv('vix-daily.csv',parse_dates=[0],index_col=0)
checks=[('1999-01-04','sp',1228.10),('2001-09-17','sp',1038.77),('2008-10-10','sp',899.22),('2018-12-31','sp',2506.85),
        ('2008-10-10','vix',69.95),('2008-11-20','vix',80.86),('2015-08-24','vix',40.74),('2018-12-31','vix',25.42)]
for d,s,v in checks:
    x=float(sp.loc[d,'Close']) if s=='sp' else float(vx.loc[d,'CLOSE'])
    print(d,s,round(x,2),'published',v,'match' if abs(x-v)<0.02 else 'MISMATCH')
