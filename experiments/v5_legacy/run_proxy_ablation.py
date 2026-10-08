"""Reproducible paired RCS failure-accounting sensitivity on original author simulator.
Uses identical designs, configuration seeds, attempt events and calibration draws
between the two literature-informed attribution variants; no carrier measurements.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
from run_original_corrected import draw_cfg, cal_energy, with_cal, sum_events, choose
import reti_sim_epoch_corrected as sim


def ablation_cfg(cfg, cal, ff, proxy):
    c=with_cal(cfg,cal,ff)
    if not proxy: c.pop('failfrac',None)
    return c

def bootstrap_cluster(data, columns, group_cols=['design','id'], reps=2000, seed=20261008):
    df=data.copy()
    gr=df.groupby(group_cols)[columns].agg(['sum','count'])
    # each group contributes paired kappa experiments; counts vary if missing
    sums=np.column_stack([gr[(x,'sum')].to_numpy(float) for x in columns])
    counts=gr[(columns[0],'count')].to_numpy(float)
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(sums),size=(reps,len(sums)))
    den=counts[idx].sum(axis=1)
    boot=sums[idx].sum(axis=1)/den[:,None]
    orig=sums.sum(axis=0)/counts.sum()
    return {c:dict(mean=float(orig[j]),lo=float(np.quantile(boot[:,j],.025)),hi=float(np.quantile(boot[:,j],.975))) for j,c in enumerate(columns)}


def sweep():
    rng=np.random.default_rng(20261007);eng=np.random.default_rng(20261021)
    cfgs=[]
    for i in range(1200):
        c=draw_cfg(rng);inn=sim.innovations(rng,4096);ec,ff=cal_energy(eng)
        cfgs.append((i,c,inn,ec,ff))
    rows=[]
    for rho in (0.,.5,.85):
        for i,c,inn,ec,ff in cfgs:
            for proxy in (False,True):
                cfg=ablation_cfg(c,ec,ff,proxy)
                s=sum_events(cfg,inn,rho)
                a=choose(s,cfg['tau'],'RETI');b=choose(s,cfg['tau'],'minE')
                rows.append(dict(id=i,rho=rho,proxy='establishment' if proxy else 'duration',
                    RETI=a,minE=b,feasible=a is not None,
                    Q_RETI=100*s[a]['Q'] if a else np.nan,Q_minE=100*s[b]['Q'] if b else np.nan,
                    E_RETI=s[a]['E'] if a else np.nan,E_minE=s[b]['E'] if b else np.nan))
        print('AB SWEEP',rho,flush=True)
    df=pd.DataFrame(rows);df.to_csv(H/'ablation_sweep_per_configuration.csv',index=False)
    summaries=[]
    for rho in (.5,):
        sub=df[df.rho==rho]; left=sub[sub.proxy=='duration'].set_index('id');right=sub[sub.proxy=='establishment'].set_index('id')
        changed=int((left.RETI.fillna('None')!=right.RETI.fillna('None')).sum())
        changedmin=int((left.minE.fillna('None')!=right.minE.fillna('None')).sum())
        for mode,d in [('duration',left),('establishment',right)]:
            feas=d[d.feasible]; diff=feas[feas.RETI!=feas.minE]
            summaries.append(dict(rho=rho,model=mode,n_total=len(d),n_feasible=len(feas),diff=len(diff),
                disagreement_pct=100*len(diff)/len(feas),
                changed_RETI_vs_other=changed,changed_minE_vs_other=changedmin,
                aggregate_dQ_pp=(feas.Q_RETI-feas.Q_minE).mean(),
                aggregate_premium_pct=(100*(feas.E_RETI/feas.E_minE-1)).mean()))
    pd.DataFrame(summaries).to_csv(H/'Table_XIII_proxy_ablation_sweep.csv',index=False)
    return df


def holdout():
    rng=np.random.default_rng(20261011);eng=np.random.default_rng(20261022)
    records=[]
    for design in ('Uniform','Copula'):
        cfgs=[]
        for i in range(600):
            c=draw_cfg(rng,copula=design=='Copula');tr=sim.innovations(rng,1024);va=sim.innovations(rng,2048)
            ec,ff=cal_energy(eng)
            cfgs.append((i,c,tr,va,ec,ff))
        for kappa in (0.,.7):
            for rho in (.5,):
                for i,c,tr,va,ec,ff in cfgs:
                    for proxy in (False,True):
                        cfg=ablation_cfg(c,ec,ff,proxy)
                        s=sum_events(cfg,tr,rho,kappa)
                        v=sum_events(cfg,va,rho,kappa)
                        a=choose(s,cfg['tau'],'RETI');b=choose(s,cfg['tau'],'minE')
                        records.append(dict(design=design,id=i,kappa=kappa,rho=rho,
                            model='establishment' if proxy else 'duration',RETI=a,minE=b,
                            feasible=a is not None, Q_RETI=100*v[a]['Q'] if a else np.nan,
                            Q_minE=100*v[b]['Q'] if b else np.nan,
                            E_RETI=v[a]['E'] if a else np.nan,E_minE=v[b]['E'] if b else np.nan,
                            viol_RETI=float(v[a]['Q']<cfg['tau']) if a else np.nan,
                            viol_minE=float(v[b]['Q']<cfg['tau']) if b else np.nan))
            print('AB HOLDOUT',design,kappa,'complete',flush=True)
    df=pd.DataFrame(records);df.to_csv(H/'ablation_holdout_per_configuration.csv',index=False)
    summaries=[]
    for rho in (.5,):
        sub=df[df.rho==rho]
        for mode in ('duration','establishment'):
            x=sub[(sub.model==mode)&sub.feasible].copy()
            x['dQ']=x.Q_RETI-x.Q_minE
            x['premium']=100*(x.E_RETI/x.E_minE-1)
            x['dViol']=x.viol_RETI-x.viol_minE
            stat=bootstrap_cluster(x,['dQ','premium','dViol'])
            summaries.append(dict(rho=rho,model=mode,n=len(x),configs=x.groupby(['design','id']).ngroups,
                diff=int((x.RETI!=x.minE).sum()),
                dQ_pp=stat['dQ']['mean'],dQ_CI_low=stat['dQ']['lo'],dQ_CI_high=stat['dQ']['hi'],
                premium_pct=stat['premium']['mean'],premium_CI_low=stat['premium']['lo'],premium_CI_high=stat['premium']['hi'],
                violation_diff=stat['dViol']['mean']))
    pd.DataFrame(summaries).to_csv(H/'Table_XIII_proxy_ablation_holdout.csv',index=False)
    # compare the same configurations for the two proxy modes
    comps=[]
    for rho in (.5,):
        p=df[df.rho==rho].pivot(index=['design','id','kappa'],columns='model',values=['RETI','minE','Q_RETI','Q_minE','E_RETI','E_minE'])
        ok=p[[('RETI','duration'),('RETI','establishment')]].notna().all(axis=1)
        s=p.loc[ok]
        r_changed=(s[('RETI','duration')]!=s[('RETI','establishment')]).sum()
        m_changed=(s[('minE','duration')]!=s[('minE','establishment')]).sum()
        comps.append(dict(rho=rho,n_matched=len(s),RETI_switches=int(r_changed),minE_switches=int(m_changed),
            RETI_switch_pct=100*r_changed/len(s)))
    pd.DataFrame(comps).to_csv(H/'ablation_holdout_route_changes.csv',index=False)
    return df

if __name__=='__main__':
    holdout()
    print(pd.read_csv(H/'Table_XIII_proxy_ablation_sweep.csv').round(3).to_string(index=False))
    print(pd.read_csv(H/'Table_XIII_proxy_ablation_holdout.csv').round(3).to_string(index=False))
    print(pd.read_csv(H/'ablation_holdout_route_changes.csv').round(3).to_string(index=False))
