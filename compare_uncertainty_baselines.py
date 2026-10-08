"""Postprocess saved matched-event holdout, comparing new methods with Wilson baseline."""
from pathlib import Path
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent

def compare(path:Path, out:Path, B=2000):
    dat=pd.read_csv(path).fillna({'selected':''})
    keys=['design','id','kappa','cluster']
    outrows=[]
    for en in ['U','L']:
        d=dat[dat.energy==en]
        for pol in ['Robust-LCB','Certified-RETI']:
            ref='Wilson-minE'
            subset=d[d.policy.isin([pol,ref])].pivot(index=keys,columns='policy',values=['selected','validation_Q_pct','validation_E','validation_violation'])
            subset=subset[(subset['selected'][pol]!='') & (subset['selected'][ref]!='')]
            if subset.empty:
                outrows.append(dict(energy=en,policy=pol,baseline=ref,matched_n=0))
                continue
            z=pd.DataFrame({'cluster':subset.index.get_level_values('cluster').values,
               'dQ':(subset[('validation_Q_pct',pol)]-subset[('validation_Q_pct',ref)]).astype(float).values,
               'dE':(subset[('validation_E',pol)]-subset[('validation_E',ref)]).astype(float).values,
               'dViol':(subset[('validation_violation',pol)]-subset[('validation_violation',ref)]).astype(float).values})
            g=z.groupby('cluster',sort=True).agg(n=('dQ','size'),dQ=('dQ','sum'),dE=('dE','sum'),dViol=('dViol','sum')).to_numpy(float)
            ix=np.random.default_rng(20261231).integers(0,len(g),size=(B,len(g)))
            nb=g[ix,0].sum(axis=1)
            row=dict(energy=en,policy=pol,baseline=ref,matched_n=len(z),clusters=len(g))
            for j,name in enumerate(['Q_pp','E_units','violation_rate'],start=1):
                b=g[ix,j].sum(axis=1)/nb
                row['d'+name]=float(g[:,j].sum()/g[:,0].sum())
                row[name+'_CI_low']=float(np.quantile(b,.025))
                row[name+'_CI_high']=float(np.quantile(b,.975))
            outrows.append(row)
    pd.DataFrame(outrows).to_csv(out,index=False)
    return pd.DataFrame(outrows)

if __name__=='__main__':
    r=compare(HERE/'results'/'holdout_extended_per_policy.csv',HERE/'results'/'paired_against_Wilson.csv')
    print(r.to_string(index=False))
