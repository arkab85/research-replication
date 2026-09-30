"""Build weekly S&P 500 log returns and log VIX changes (Friday closes), 1999-2018, as used in Section 9."""
import pandas as pd, numpy as np
sp=pd.read_csv('sp500.csv.gz',parse_dates=[0],index_col=0)['Adj Close']
vx=pd.read_csv('vix-daily.csv',parse_dates=[0],index_col=0)['CLOSE']
d=pd.concat([sp.rename('sp'),vx.rename('vix')],axis=1,sort=True).dropna().loc['1999-01-01':'2018-12-31']
w=d.resample('W-FRI').last().dropna()
y=pd.DataFrame({'ret':100*np.log(w.sp).diff(),'dvix':100*np.log(w.vix).diff()}).dropna()
y.to_csv('weekly.csv'); print(len(y))
