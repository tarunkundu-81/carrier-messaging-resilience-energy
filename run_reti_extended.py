#!/usr/bin/env python3
"""Experimental extension of the recovered RETI code; original files unmodified.

At rho=0.5 the 600x2-design holdout is reproduced using seeds and procedures
from run_original_corrected.py. Two additional uncertainty-aware policies are
compared with the existing six-action policies and independent MC reference.
Does NOT claim empirical carrier calibration. Deterministic outputs go to --out.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from reti_uncertainty import joint_intervals,decide,disagreement_margin
from reti_sim_epoch_corrected import ACTIONS,run_actions,innovations,choose as choose_base,summarize
from run_original_corrected import draw_cfg,with_cal,cal_energy,choose as choose_prior

POLICIES=("RETI","minE","maxQ","W5","W20","Knee","Wilson-minE", "MC-reference", "Robust-LCB", "Certified-RETI")

def labels(model,cfg,inn,rho,kappa):
    raw=run_actions(cfg,inn,rho,kappa=kappa,eq16=True)
    return raw,summarize(raw,cfg['L'])

def run_holdout(out:Path,ncfg=600,ntr=1024,nva=2048,noracle=8192,rho=0.5,seed=20261011, energy_seed=20261022, confidence=.95):
    rng=np.random.default_rng(seed); rngE=np.random.default_rng(energy_seed)
    rows=[]; bounds_rows=[]; progress=[]
    for design in ('Uniform','Copula'):
        # Original loop order matters: draw cfg, training, validation, then calibration.
        cfgs=[]
        for i in range(ncfg):
            c=draw_cfg(rng,copula=(design=='Copula'))
            tr=innovations(rng,ntr); va=innovations(rng,nva)
            ec,ff=cal_energy(rngE)
            # independent reference uses an independent seed (as in original script)
            oracleinn=innovations(np.random.default_rng(20261215+100000*(design=='Copula')+i),noracle)
            cfgs.append((c,with_cal(c,ec,ff),tr,va,oracleinn))
        for kappa in (0.,.7):
            for i,(cu,cl,tr,va,reference) in enumerate(cfgs):
                for energy,cfg in (('U',cu),('L',cl)):
                    train_raw,a=labels(None,cfg,tr,rho,kappa)
                    _,b=labels(None,cfg,va,rho,kappa)
                    _,truth=labels(None,cfg,reference,rho,kappa)
                    intervals=joint_intervals(train_raw,confidence=confidence)
                    assert set(intervals)==set(ACTIONS)
                    certified,status_c,extra=decide(intervals,cfg['tau'],'Certified-RETI')
                    robust,status_r,_=decide(intervals,cfg['tau'],'Robust-LCB')
                    selected={p:choose_prior(a,cfg['tau'],p,ntr) for p in POLICIES[:7]}
                    selected.update({'MC-reference':choose_prior(truth,cfg['tau'],'minE',noracle),
                                     'Robust-LCB':robust,'Certified-RETI':certified})
                    for p in POLICIES:
                        ch=selected[p]
                        rows.append(dict(design=design,id=i,cluster=f'{design}-{i}',kappa=kappa,
                          rho=rho,energy=energy,tau=cfg['tau'],L_ms=cfg['L'],policy=p,
                          selected=ch or '',decides=ch is not None,
                          status=(status_c if p=='Certified-RETI' else status_r if p=='Robust-LCB' else 'point'),
                          safe_count=extra['safe_count'],possible_count=extra['possible_count'],
                          validation_Q_pct=100*b[ch]['Q'] if ch else np.nan,
                          validation_E=b[ch]['E'] if ch else np.nan,
                          validation_violation=int(b[ch]['Q']<cfg['tau']) if ch else np.nan,
                          reference_Q_pct=100*truth[ch]['Q'] if ch else np.nan,
                          reference_E=truth[ch]['E'] if ch else np.nan))
                    # Per-action diagnostics on nominal simultaneous bounds
                    for action,iv in intervals.items():
                        bounds_rows.append(dict(design=design,id=i,kappa=kappa,energy=energy,
                                                action=action,qhat=iv.qhat, q_lower=iv.qlo, q_upper=iv.qhi,
                                                ehat=iv.ehat,e_lower=iv.elo,e_upper=iv.ehi,tau=cfg['tau']))
            progress.append(dict(design=design,kappa=kappa,complete=True))
            print('HOLDOUT',design,kappa,'completed',flush=True)
    dat=pd.DataFrame(rows)
    dat.to_csv(out/'holdout_extended_per_policy.csv',index=False)
    pd.DataFrame(bounds_rows).to_csv(out/'holdout_interval_diagnostics.csv',index=False)
    assert len(dat)==2*ncfg*2*2*len(POLICIES)
    tab=dat.groupby(['energy','policy'],sort=True,observed=True).agg(
        total=('decides','size'),served=('decides','sum'),
        Q_pct=('validation_Q_pct','mean'),mean_E=('validation_E','mean'),
        violations=('validation_violation','sum')).reset_index()
    tab['abstain']=tab.total-tab.served
    tab['coverage_pct']=100*tab.served/tab.total
    tab['violation_rate_given_served_pct']=100*tab.violations/tab.served.replace(0,np.nan)
    tab.to_csv(out/'policy_coverage_and_outcomes.csv',index=False)

    # Common match: all policies that require confident selection; don't hide abstentions.
    keys=['design','id','kappa','energy']
    wide=dat.pivot(index=keys,columns='policy',values='selected')
    common=wide[(wide!='').all(axis=1)]
    common.to_csv(out/'all_policy_common_decisions.csv')
    cm=dat.merge(common.reset_index()[keys],on=keys,how='inner')
    cm.groupby(['energy','policy'],sort=True).agg(n=('decides','size'),
      Q_pct=('validation_Q_pct','mean'),mean_E=('validation_E','mean'),
      violations=('validation_violation','sum')).reset_index().to_csv(out/'all_policy_matched_results.csv',index=False)

    # Every comparison uses a matched cohort for just the two policies.
    comps=[]; paired=[]
    for en in ('U','L'):
        d=dat[dat.energy==en]
        for pol in ('Robust-LCB','Certified-RETI','Wilson-minE','RETI'):
            t=d[d.policy.isin([pol,'minE'])].pivot(index=['design','id','kappa','cluster'],columns='policy',
                                   values=['selected','validation_Q_pct','validation_E','validation_violation'])
            if pol=='minE':continue
            keep=t[(t[('selected',pol)]!='') & (t[('selected','minE')]!='')].copy()
            if len(keep)==0:
                comps.append(dict(energy=en,policy=pol,matched_n=0,clusters=0,
                                  delta_Q_pp=np.nan,Q_lo=np.nan,Q_hi=np.nan,
                                  delta_E_units=np.nan,E_lo=np.nan,E_hi=np.nan,
                                  delta_violation=np.nan,viol_lo=np.nan,viol_hi=np.nan))
                continue
            z=pd.DataFrame({
              'dQ':(keep[('validation_Q_pct',pol)]-keep[('validation_Q_pct','minE')]).astype(float),
              'dE':(keep[('validation_E',pol)]-keep[('validation_E','minE')]).astype(float),
              'dViolation':(keep[('validation_violation',pol)]-keep[('validation_violation','minE')]).astype(float),
              'cluster':keep.index.get_level_values('cluster')}).reset_index(drop=True)
            sums=z.groupby('cluster',sort=True).agg(n=('dQ','size'),dQ=('dQ','sum'),dE=('dE','sum'),dViolation=('dViolation','sum'))
            mat=sums.to_numpy(float)
            B=2000
            rngb=np.random.default_rng(20261220)
            idx=rngb.integers(0,len(sums),size=(B,len(sums)))
            denominator=mat[idx,0].sum(axis=1)
            o={'energy':en,'policy':pol,'matched_n':len(z),'clusters':len(sums)}
            for j,(p,label) in enumerate((('dQ','Q'),('dE','E'),('dViolation','viol')),start=1):
                boot=mat[idx,j].sum(axis=1)/denominator
                o['delta_'+{'Q':'Q_pp','E':'E_units','viol':'violation'}[label]]=float(mat[:,j].sum()/mat[:,0].sum())
                o[label+'_lo']=float(np.quantile(boot,.025));o[label+'_hi']=float(np.quantile(boot,.975))
            comps.append(o)
    pd.DataFrame(comps).to_csv(out/'matched_paired_bootstrap.csv',index=False)
    assert ((tab.served + tab.abstain)==tab.total).all()
    return tab,dat


def run_boundary(out:Path):
    """Analyze the verified original sweep WITHOUT interpreting draws as deployment prevalence."""
    src=out/'prior_verified_sweep_per_configuration.csv'
    if not src.exists(): src=Path(__file__).resolve().parent/'results'/'prior_verified_sweep_per_configuration.csv'
    df=pd.read_csv(src)
    # Draw configurations with exact original sampling order: 4096 innovations between draws.
    rng=np.random.default_rng(20261007)
    deadline=[]
    for i in range(1200):
        c=draw_cfg(rng)
        deadline.append(c['L'])
        _=innovations(rng,4096)
    df['deadline_ms']=df.id.map(dict(enumerate(deadline)))
    rows=[]
    for (rho,en),grp in df.groupby(['rho','energy']):
        feasible=grp[grp.feasible.astype(bool)].copy()
        feasible['diff']=feasible.RETI!=feasible.minE
        feasible['reliability_gain_pp']=100*(feasible.Q_RETI-feasible.Q_minE)
        feasible['energy_premium_pct']=100*(feasible.E_RETI/feasible.E_minE-1)
        for tert,subset in [('all',feasible),*[(f'tertile_{i+1}',feasible.loc[feasible.deadline_ms.between(lo,hi,inclusive='right' if i else 'both')]) for i,(lo,hi) in enumerate(zip([-float('inf')]+list(np.quantile(deadline,[1/3,2/3])),list(np.quantile(deadline,[1/3,2/3]))+[float('inf')]))]]:
            nz=subset[subset['diff']]
            rows.append(dict(rho=rho,energy=en,deadline_bin=tert,feasible=len(subset),
                disagreements=len(nz),disagreement_pct=100*len(nz)/len(subset) if len(subset) else np.nan,
                conditional_gain_pp=nz.reliability_gain_pp.mean(),
                conditional_premium_pct=nz.energy_premium_pct.mean()))
        # Identity check for all disagreements: gain > 0 and premium >=0
        nz=feasible[feasible['diff']]
        assert (nz.reliability_gain_pp>0).all()
        assert (nz.energy_premium_pct>0).all()
    result=pd.DataFrame(rows)
    result.to_csv(out/'policy_decision_boundary_by_deadline.csv',index=False)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--configs',type=int,default=600,help='Holdout configurations per design (600 original)')
    parser.add_argument('--train',type=int,default=1024)
    parser.add_argument('--validation',type=int,default=2048)
    parser.add_argument('--reference',type=int,default=8192)
    parser.add_argument('--confidence',type=float,default=.95)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    # original holdout n=600, seed fixed, other numbers are explicitly user-adjustable
    boundary=run_boundary(args.out)
    tab,data=run_holdout(args.out,ncfg=args.configs,ntr=args.train,nva=args.validation,noracle=args.reference,confidence=args.confidence)
    cfg=dict(configs_per_design=args.configs,train=args.train,validation=args.validation,
        mc_reference=args.reference,confidence=args.confidence, rho=.5,
        input_source='Original corrected six-action simulator; author-constructed synthetic input priors',
        baseline_source='Original 1200-config sweep results from verified package',
        restrictions=['No real carrier calibration','Nominal simultaneous energy CI relies on t approximation',
          'Strict certification may abstain frequently','MC-reference is approximate and not an omniscient oracle'])
    (args.out/'run_metadata.json').write_text(json.dumps(cfg,indent=2))
    print(tab[['energy','policy','served','abstain','Q_pct','violations']].to_string(index=False,float_format=lambda v:f'{v:.3f}'))
    print('BOUNDARY')
    print(boundary[boundary.deadline_bin=='all'][['rho','energy','feasible','disagreements','disagreement_pct']].to_string(index=False))

if __name__=='__main__':main()
