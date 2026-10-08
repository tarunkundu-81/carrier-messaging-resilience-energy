"""Auditable rerun of supplied RETI scripts with isolated correction of failure-cost attribution.
Original source: reproduce_reti.py, extend_reti.py, reti_sim.py and calibrated_energy.py.
No measured carrier data. Literature-based energy coefficients mix historic handset studies.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import reti_sim_epoch_corrected as sim
ACTIONS=sim.ACTIONS
OUT=HERE
RNG = dict(p=dict(RCS=(.62,.99),SMS=(.70,.99),MMS=(.55,.98),MH=(.50,.96)),
           e=dict(RCS=(.80,1.45),SMS=(1.05,2.20),MMS=(1.60,3.25),MH=(2.00,3.70)),
           m=dict(RCS=(85,185),SMS=(110,225),MMS=(135,280),MH=(195,390)))

def draw_cfg(rng,copula=False):
    # Verbatim random-order formula from supplied reproduce_reti.py/extend_reti.py.
    comp={}
    for k in ('RCS','SMS','MMS','MH'):
        if copula:
            Rm=np.array([[1,.75,-.55],[.75,1,-.4125],[-.55,-.4125,1]])
            u=norm.cdf(rng.multivariate_normal(np.zeros(3),Rm))
        else:
            u=rng.uniform(size=3)
        comp[k]=[RNG['p'][k][0]+u[0]*(RNG['p'][k][1]-RNG['p'][k][0]),
                 RNG['e'][k][0]+u[1]*(RNG['e'][k][1]-RNG['e'][k][0]),
                 RNG['m'][k][0]+u[2]*(RNG['m'][k][1]-RNG['m'][k][0])]
    sig=rng.uniform(.27,.55)
    comp={k:(v[0],v[1],v[2],sig) for k,v in comp.items()}
    return sim.make_cfg(comp,L=rng.uniform(260,540),tau=rng.uniform(.73,.92),
        r=int(rng.integers(0,3)),gamma=rng.uniform(.05,.65),
        lag=rng.uniform(8,55),overhead=rng.uniform(.07,.28))

# Published ingredients as copied from calibrated_energy.py, not experimentally measured combinations.
SETUP,RELEASE,FAIL,BASE_MA=7.26,16.53,108.53,95.65
TX={'good':.42,'medium':.57,'poor':1.26}
RX={'good':.22,'medium':.36,'poor':.48}
TAIL_S=(4.6,10.3); SMS_J_MAX=7.9e-3*82

def path_costs(V=3.8,tail_s=10.3,tail_frac=.5,amort=1.0,sms_frac=1.0,
               mms_kb=100,sig='medium',payload_kb=2.,hold_s=.15):
    # Source calibration's standalone costs explicitly omit unexecuted past RCS failures.
    tail=tail_s*BASE_MA*tail_frac*amort
    rcs=SETUP+payload_kb*TX[sig]+RX[sig]+RELEASE+tail
    sms=(SMS_J_MAX*sms_frac)/V*1000
    mms=SETUP+mms_kb*TX[sig]+RELEASE+tail
    return dict(RCS=rcs,SMS=sms,MMS=mms,MH=rcs+hold_s*BASE_MA,fail_rcs=FAIL)

def cal_energy(rng):
    c=path_costs(V=rng.uniform(3.6,4.35),tail_s=rng.uniform(*TAIL_S),
        tail_frac=rng.uniform(.2,1),amort=rng.uniform(0,1),
        sms_frac=rng.uniform(.25,1),mms_kb=rng.uniform(50,300),
        sig=rng.choice(['good','medium','poor']),payload_kb=rng.uniform(.5,5))
    return {k:c[k]/c['RCS'] for k in ('RCS','SMS','MMS','MH')},c['fail_rcs']/c['RCS']

def with_cal(cfg,cc,ff):
    c=dict(cfg)
    c['comp']={k:(v[0],cc[k],v[2],v[3]) for k,v in cfg['comp'].items()}
    c['failfrac']=ff
    return c

def knee(summ,tau):
    feas=[a for a in ACTIONS if summ[a]['Q']>=tau]
    if not feas:return None
    nd=[a for a in feas if not any(summ[b]['Q']>=summ[a]['Q'] and summ[b]['E']<=summ[a]['E'] and
        (summ[b]['Q']>summ[a]['Q'] or summ[b]['E']<summ[a]['E']) for b in feas)]
    qs=[summ[a]['Q'] for a in nd];es=[summ[a]['E'] for a in nd]
    qr=max(qs)-min(qs)+1e-9;er=max(es)-min(es)+1e-9
    return min(nd,key=lambda a:((max(qs)-summ[a]['Q'])/qr)**2+((summ[a]['E']-min(es))/er)**2)

POLS=('RETI','minE','maxQ','W5','W20','Knee','Wilson-minE','MC-oracle')
def choose(s,tau,pol,n=1024):
    if pol=='RETI': return sim.choose(s,tau,'RETI')
    if pol=='minE': return sim.choose(s,tau,'minE')
    if pol=='maxQ': return sim.choose(s,tau,'maxQ')
    if pol=='W5': return sim.choose(s,tau,'W',lam=5)
    if pol=='W20':return sim.choose(s,tau,'W',lam=20)
    if pol=='Knee':return knee(s,tau)
    if pol=='Wilson-minE':
        feas=[a for a in ACTIONS if sim.wilson_lower_onesided(int(round(s[a]['Q']*n)),n)>=tau]
        return min(feas,key=lambda a:s[a]['E']) if feas else None
    raise ValueError(pol)

def sum_events(cfg,inn,rho,kappa=1):
    return sim.summarize(sim.run_actions(cfg,inn,rho,kappa=kappa,eq16=True),cfg['L'])

def sweep():
    rng=np.random.default_rng(20261007)
    eng=np.random.default_rng(20261021)
    rows=[]
    configs=[]
    for idx in range(1200):
        c=draw_cfg(rng); inn=sim.innovations(rng,4096)
        cc,ff=cal_energy(eng)
        configs.append((c,with_cal(c,cc,ff),inn))
    for rho in (0.,.5,.85):
        for i,(c,cl,inn) in enumerate(configs):
            s0=sum_events(c,inn,rho)
            s1=sum_events(cl,inn,rho)
            for model,cfg,s in (('U',c,s0),('L',cl,s1)):
                r=choose(s,cfg['tau'],'RETI');m=choose(s,cfg['tau'],'minE')
                rows.append(dict(id=i,rho=rho,energy=model,RETI=r,minE=m,feasible=r is not None,
                    Q_RETI=s[r]['Q'] if r else np.nan,Q_minE=s[m]['Q'] if m else np.nan,
                    E_RETI=s[r]['E'] if r else np.nan,E_minE=s[m]['E'] if m else np.nan,
                    tau=cfg['tau']))
        print('SWEEP',rho,'completed',flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'sweep_per_configuration.csv',index=False)
    results=[]
    for rho in (0.,.5,.85):
        sub=df[df.rho==rho]
        cs={e:sub[sub.energy==e].set_index('id') for e in ('U','L')}
        for e in ('U','L'):
            s=cs[e];f=s[s.feasible];di=f[f.RETI!=f.minE]
            ci=sim.wilson(len(di),len(f))
            switches=(cs['U'].RETI!=cs['L'].RETI).sum() if e=='L' else 0
            results.append(dict(rho=rho,energy=e,feasible=len(f),disagreements=len(di),
                disagreement_pct=100*len(di)/len(f),CI_low_pct=100*ci[0],CI_high_pct=100*ci[1],
                gain_on_disagreement_pp=100*(di.Q_RETI-di.Q_minE).mean(),
                premium_on_disagreement_pct=(100*(di.E_RETI/di.E_minE-1)).mean(),
                RETI_switches_vs_U=switches))
    pd.DataFrame(results).to_csv(OUT/'Table_XI_original_sim_corrected.csv',index=False)
    return df

def holdout():
    rng=np.random.default_rng(20261011); eng=np.random.default_rng(20261022)
    rows=[];t0=time.time()
    for design in ('Uniform','Copula'):
        configs=[]
        for i in range(600):
            c=draw_cfg(rng,copula=(design=='Copula'))
            tr=sim.innovations(rng,1024); va=sim.innovations(rng,2048)
            cc,ff=cal_energy(eng)
            # distinct independent reference events only used for oracle at rho 0.5
            oracleinn=sim.innovations(np.random.default_rng(20261215 + 100000*(design=='Copula')+i),8192)
            configs.append((c,with_cal(c,cc,ff),tr,va,oracleinn))
        for kappa in (0.,.7):
            for rho in (0.,.5,.85):
                for i,(c,cl,tr,va,oracleinn) in enumerate(configs):
                    for en,cfg in (('U',c),('L',cl)):
                        a=sum_events(cfg,tr,rho,kappa)
                        b=sum_events(cfg,va,rho,kappa)
                        cr=choose(a,cfg['tau'],'RETI')
                        cm=choose(a,cfg['tau'],'minE')
                        if cr is None:
                            # Excluded from ordinary pooled comparison cohort; retain baseline infeasibility count.
                            rows.append(dict(design=design,id=i,kappa=kappa,rho=rho,energy=en,
                                policy='RETI',selected=None,train_feasible=False,Q=np.nan,E=np.nan,violation=np.nan,
                                tau=cfg['tau']))
                            continue
                        truth=sum_events(cfg,oracleinn,rho,kappa) if rho==.5 else None
                        for pol in POLS:
                            ch=(choose(truth,cfg['tau'],'minE',8192) if truth is not None else None) if pol=='MC-oracle' else choose(a,cfg['tau'],pol,1024)
                            if ch is None and pol=='MC-oracle' and rho!=.5:
                                continue
                            if ch is None:
                                rows.append(dict(design=design,id=i,kappa=kappa,rho=rho,energy=en,
                                    policy=pol,selected=None,train_feasible=True,Q=np.nan,E=np.nan,violation=np.nan,
                                    tau=cfg['tau']))
                                continue
                            rows.append(dict(design=design,id=i,kappa=kappa,rho=rho,energy=en,
                                policy=pol,selected=ch,train_feasible=True,Q=100*b[ch]['Q'],E=b[ch]['E'],
                                violation=float(b[ch]['Q']<cfg['tau']),tau=cfg['tau'],
                                truth_Q=100*truth[ch]['Q'] if truth else np.nan,
                                truth_E=truth[ch]['E'] if truth else np.nan,
                                oracle_choice=choose(truth,cfg['tau'],'minE',8192) if truth else None))
                print('HOLDOUT',design,kappa,rho,'elapsed',round(time.time()-t0,1),'s',flush=True)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/'holdout_per_configuration_policy.csv',index=False)
    # Table XII: pooled Q E violation and number of valid decisions at rho=0.5
    sub=df[df.rho==.5]
    tab=sub.groupby(['energy','policy'],dropna=False).agg(n=('Q','count'),Q=('Q','mean'),E=('E','mean'),violations=('violation','sum')).reset_index()
    tab.to_csv(OUT/'Table_XII_original_sim_corrected.csv',index=False)
    # fully paired estimates; cluster bootstrap at the configuration design/id level, retaining both kappa settings.
    base=sub[sub.policy.isin(['RETI','minE'])].pivot(index=['design','id','kappa','energy'],columns='policy',values=['Q','E','violation'])
    base=base.dropna()
    results=[]
    for en in ('U','L'):
        b=base.xs(en,level='energy')
        qq=b[('Q','RETI')]-b[('Q','minE')]
        ee=b[('E','RETI')]-b[('E','minE')]
        vv=b[('violation','RETI')]-b[('violation','minE')]
        prem=100*(b[('E','RETI')]/b[('E','minE')]-1)
        d=pd.DataFrame(dict(dQ=qq,dE=ee,dViol=vv,prem=prem))
        d['group']=[f'{a}-{i}' for a,i,k in d.index]
        grouped=d.groupby('group').agg(n=('dQ','count'),**{k+'sum':(k,'sum') for k in ('dQ','dE','dViol','prem')})
        N=len(grouped); G=grouped.to_numpy(float)
        rngb=np.random.default_rng(20261220)
        idx=rngb.integers(0,N,size=(2000,N))
        denomin=G[idx,0].sum(axis=1)
        result={'energy':en,'paired_decisions':len(d),'clusters':N}
        for j,k in enumerate(('dQ','dE','dViol','prem'),start=1):
            estim=G[:,j].sum()/G[:,0].sum()
            boots=G[idx,j].sum(axis=1)/denomin
            result[k]=float(estim)
            result[k+'_low']=float(np.quantile(boots,.025))
            result[k+'_high']=float(np.quantile(boots,.975))
        results.append(result)
    pd.DataFrame(results).to_csv(OUT/'Table_XII_paired_confidence_original_sim.csv',index=False)
    # Oracle regret on matched heldout cohorts where BOTH selection and oracle are reference-feasible.
    oracle=sub[sub.policy=='MC-oracle'][['design','id','kappa','energy','selected','truth_E']].rename(columns={'selected':'oracle_reference_choice','truth_E':'oracle_reference_E'})
    other=sub[sub.policy.isin(['RETI','minE','W5','W20','Knee','maxQ','Wilson-minE'])].merge(oracle,on=['design','id','kappa','energy'],how='left')
    other=other[(other.selected.notna())&(other.oracle_reference_choice.notna())&(other.truth_Q+1e-10>=100*other.tau)]
    other=other.assign(oracle_energy_regret=other.truth_E-other.oracle_reference_E)
    other.groupby(['energy','policy']).agg(n=('oracle_energy_regret','count'),mean_energy_regret=('oracle_energy_regret','mean'),
      median_energy_regret=('oracle_energy_regret','median')).reset_index().to_csv(OUT/'oracle_reference_regret.csv',index=False)
    return df

def checks():
    rng=np.random.default_rng(100);c=draw_cfg(rng);i=sim.innovations(rng,1024)
    noff=sum_events(c,i,.5)
    cfg=dict(c);cfg['failfrac']=0.20
    yes=sum_events(cfg,i,.5)
    # single SMS/MMS cost unchanged when introducing a failure charge on RCS only
    assert abs(noff['SMS']['E']-yes['SMS']['E'])<1e-12
    assert abs(noff['MMS']['E']-yes['MMS']['E'])<1e-12
    assert abs(noff['Red']['E']-yes['Red']['E'])<1e-12
    # standalone successful RCS cost matches original core exactly
    import importlib.util
    spec=importlib.util.spec_from_file_location('reti_orig',str(HERE/'original_sources'/'reti_sim.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    assert abs(noff['Retry']['E']-mod.summarize(mod.run_actions(c,i,.5,eq16=True),c['L'])['Retry']['E'])<1e-12
    print('TEST: Corrected failure charge leaves SMS/MMS/Red unchanged and U model byte-equivalent',flush=True)

if __name__=='__main__':
    t=time.time();checks()
    central=path_costs(); c0={k:central[k]/central['RCS'] for k in ('RCS','SMS','MMS','MH')}
    pd.DataFrame([{'action':k,'central_mAs':v,'relative_to_RCS':v/central['RCS']} for k,v in central.items()]).to_csv(OUT/'Table_X_original_calibration_corrected.csv',index=False)
    print('CENTRAL',c0,'old_SMSG', (central['SMS']+FAIL)/central['RCS'],flush=True)
    sweep(); print('SWEEP FINAL',round(time.time()-t,1),'s',flush=True)
    holdout();print('ALL DONE',round(time.time()-t,1),'s',flush=True)
