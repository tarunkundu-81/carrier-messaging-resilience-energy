"""All-policy matched cohort (same configurations, same κ and held-out events) and paired cluster intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent
D=pd.read_csv(H/'holdout_per_configuration_policy.csv');D=D[D.rho==.5].copy()
policies=['RETI','minE','maxQ','W5','W20','Knee','Wilson-minE','MC-oracle']
idx=['design','id','kappa','energy']
wide=D.pivot(index=idx,columns='policy',values='selected')
assert all(p in wide.columns for p in policies)
complete=wide[policies].notna().all(axis=1)
paired=complete.unstack('energy').fillna(False).all(axis=1)
cohort=paired[paired].index
C=D.set_index(['design','id','kappa']).loc[cohort].reset_index()
C=C[C.policy.isin(policies)].copy()
assert C.groupby(['energy','policy']).size().nunique()==1
assert len(C.groupby(['design','id','kappa']).size())==len(cohort)
N=len(cohort)
C.to_csv(H/'holdout_8_policy_common_cohort.csv',index=False)
T=C.groupby(['energy','policy']).agg(n=('Q','size'),Q_pct=('Q','mean'),E=('E','mean'),violations=('violation','sum')).reset_index()
T['violations']=T.violations.astype(int)
T.to_csv(H/'Table_XII_all_policy_matched.csv',index=False)
A=D.groupby(['energy','policy']).agg(n_available=('selected','count')).reset_index();A['n_base']=2205;A['abstentions']=A.n_base-A.n_available
A.to_csv(H/'Table_XII_abstention_counts.csv',index=False)
# paired confidence intervals by configuration cluster for same evaluated cohort
P=C.pivot(index=idx,columns='policy',values=['Q','E','violation'])
R=[]
for en in ('U','L'):
    p=P.xs(en,level='energy')
    for pol in policies:
        if pol=='minE':continue
        delta=pd.DataFrame({f'd_{x}':p[(x,pol)]-p[(x,'minE')] for x in ('Q','E','violation')})
        delta['rel_E_pct']=100*(p[('E',pol)]/p[('E','minE')]-1)
        delta['cluster']=[f'{x[0]}|{x[1]}' for x in delta.index]
        G=delta.groupby('cluster').agg(n=('d_Q','size'),dQ=('d_Q','sum'),dE=('d_E','sum'),dV=('d_violation','sum'),relE=('rel_E_pct','sum')).to_numpy(float)
        seed=20261013;rg=np.random.default_rng(seed);B=2000;ii=rg.integers(0,len(G),size=(B,len(G)))
        samp=G[ii];tot=samp[:,:,0].sum(axis=1)
        for c,nm in [(1,'dQ_pp'),(2,'dE_units'),(3,'dViol_fraction'),(4,'relative_premium_pct')]:
            bs=samp[:,:,c].sum(axis=1)/tot
            R.append(dict(energy=en,policy=pol,metric=nm,n=N,clusters=len(G),estimate=float(G[:,c].sum()/G[:,0].sum()),
                CI_low=float(np.percentile(bs,2.5)),CI_high=float(np.percentile(bs,97.5))))
pd.DataFrame(R).to_csv(H/'Table_XII_common_cohort_paired_CIs.csv',index=False)
# matched oracle-reference energy regret: same cohort; all policies evaluated and oracle reference-feasible if applicable
O=C[C.policy!='MC-oracle'].copy()
OR=C[C.policy=='MC-oracle'][idx+['selected']].rename(columns={'selected':'oracle_choice'})
O=O.merge(OR,on=idx,validate='many_to_one')
O['reference_feasible']=O['truth_Q']>=O['tau']*100-1e-10
M=O[O.reference_feasible]
Z=M.groupby(['energy','policy']).agg(n=('truth_E','size'),mean_oracle_ref_regret=('truth_E','mean')).reset_index()
# derive oracle energy from oracle policy truth_E (per-case reference model)
Rref=C[C.policy=='MC-oracle'][idx+['truth_E']].rename(columns={'truth_E':'oracle_truth_E'})
M=M.merge(Rref,on=idx,validate='many_to_one');M['regret']=M.truth_E-M.oracle_truth_E
M.groupby(['energy','policy']).agg(n=('regret','size'),mean_regret=('regret','mean'),median_regret=('regret','median')).reset_index().to_csv(H/'matched_oracle_regret.csv',index=False)
print('common cohort',N, 'config clusters', C.groupby(['design','id']).ngroups)
print(T.round(5).to_string(index=False))
print('abstention',A.to_string(index=False))
print('matched confidence intervals RETI vs minE')
print(pd.DataFrame(R).query("policy=='RETI'").round(5).to_string(index=False))
