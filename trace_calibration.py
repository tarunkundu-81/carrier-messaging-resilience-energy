#!/usr/bin/env python3
"""Optional empirical trace summarizer. Does NOT inject observations into simulator.

Required CSV columns:
 action,condition,recipient_confirmed,delivery_latency_ms,deadline_ms,
 incremental_energy_mJ,measurement_source
Rows are individual dispatched messages. 'recipient_confirmed' must be 0/1;
failed cases may have a missing latency but must have a measured energy value.
This tool can only summarize real observations supplied by the researcher;
it does not infer operator measurements from synthetic events or standards.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import beta
from reti_sim_epoch_corrected import ACTIONS

REQUIRED=['action','condition','recipient_confirmed','delivery_latency_ms',
          'deadline_ms','incremental_energy_mJ','measurement_source']

def cp_bounds(k,n,alpha=.05):
    return ((float(beta.ppf(alpha/2,k,n-k+1)) if k>0 else 0.),
            (float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.))

def analyze_trace(path:Path, out:Path, min_trials:int=30):
    df=pd.read_csv(path)
    missing=set(REQUIRED)-set(df.columns)
    if missing:raise ValueError('Required columns missing: '+', '.join(sorted(missing)))
    if len(df)==0:raise ValueError('No observations')
    if df[REQUIRED].isna().any().any():
        # Only failed/unconfirmed messages may lack a completion latency.
        null=df[REQUIRED].isna().drop(columns=['delivery_latency_ms']).any(axis=1)
        if null.any():raise ValueError('Null required field other than failed-case completion latency')
    if not set(df.action).issubset(set(ACTIONS)):
        raise ValueError('Use simulator action tags: '+str(ACTIONS))
    if not set(pd.to_numeric(df.recipient_confirmed)).issubset({0,1}):
        raise ValueError('recipient_confirmed must be 0 or 1')
    for col in ('deadline_ms','incremental_energy_mJ'):
        if not np.isfinite(pd.to_numeric(df[col])).all() or not (df[col]>0).all():
            raise ValueError(f'{col} must be positive and finite')
    if ((df.recipient_confirmed==1)&(df.delivery_latency_ms.isna())).any():
        raise ValueError('Delivered messages require complete recipient delivery latency')
    if ((df.delivery_latency_ms.dropna()<0)).any():
        raise ValueError('Latency cannot be negative')
    df['before_deadline']=(df.recipient_confirmed==1)&(df.delivery_latency_ms<=df.deadline_ms)
    out_rows=[]
    for (act,cond), g in df.groupby(['action','condition'],sort=True):
        n=len(g);k=int(g.before_deadline.sum());lo,hi=cp_bounds(k,n)
        succ=g[g.recipient_confirmed==1].delivery_latency_ms.dropna()
        lns=np.log(succ[succ>0])
        out_rows.append(dict(action=act,condition=cond,n=n,success_by_deadline=k,
           Q_point=k/n,Q_CI_low=lo,Q_CI_high=hi,
           incremental_energy_mean_mJ=float(g.incremental_energy_mJ.mean()),
           incremental_energy_sd_mJ=float(g.incremental_energy_mJ.std(ddof=1)) if n>1 else np.nan,
           delivered_latency_median_ms=float(succ.median()) if len(succ) else np.nan,
           delivered_lognormal_sigma=float(lns.std(ddof=1)) if len(lns)>1 else np.nan,
           source_labels='; '.join(sorted(map(str,g.measurement_source.unique()))),
           sample_size_flag='low_n' if n<min_trials else 'ok',
           limitation='success-conditioned latency not unconditional attempt latency'))
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out,index=False)
    return pd.DataFrame(out_rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--out',type=Path,default=Path('trace_calibrated_summary.csv'))
    p.add_argument('--min-trials',type=int,default=30)
    args=p.parse_args();r=analyze_trace(args.trace,args.out,args.min_trials)
    print(f'Empirical summaries written: {len(r)} path-condition groups -> {args.out}')
