import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=json.load(open('output/results.json'))
# Fig1
f=r['fig1']; plt.figure(figsize=(4.2,3.1),dpi=200)
for k,m in zip(['Normal','Partial','Severe'],['o','s','^']): plt.plot(f['rho'],f['gain'][k],marker=m,ms=3.5,label=k)
plt.xlabel('Shared latent stress fraction (rho)'); plt.ylabel('Gain over best single component (pp)')
plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig('figs/fig1.png')
# Fig2
f=r['fig2']; names={'RCS':'RCS','SMS':'SMS','MMS':'MMS','Retry':'Retry','MH':'Multi-hop','Red':'Redundant'}
pts={k:(v['E'],v['Q'],v['p95']) for k,v in f.items()}
def dom(a,b): return b[0]<=a[0] and b[1]>=a[1] and b[2]<=a[2] and (b[0]<a[0] or b[1]>a[1] or b[2]<a[2])
nd={k for k in pts if not any(dom(pts[k],pts[j]) for j in pts if j!=k)}
plt.figure(figsize=(4.2,3.2),dpi=200)
for k,(e,q,p) in pts.items():
    feas=q>=80
    plt.scatter(e,q,s=p*0.9,c='#1f77b4' if feas else 'lightgray',edgecolors='black' if k in nd else 'gray',linewidths=1.6 if k in nd else .8,zorder=3)
    plt.annotate(names[k],(e,q),xytext=(6,5),textcoords='offset points',fontsize=7)
plt.axhline(80,ls='--',c='gray',lw=1); plt.xlabel('Mean incremental energy (relative units)'); plt.ylabel('Success by deadline (%)')
plt.xlim(.75,4.4); plt.ylim(40,95); plt.grid(alpha=.2); plt.tight_layout(); plt.savefig('figs/fig2.png')
print('nondominated',nd)
# Fig3
t=r['table9']; xs=[str(x['rho']) for x in t]; y=[x['pct'] for x in t]
lo=[x['pct']-x['ci'][0] for x in t]; hi=[x['ci'][1]-x['pct'] for x in t]; rb=[x['robust_pct'] for x in t]
plt.figure(figsize=(5.0,2.8),dpi=200)
import numpy as np
x=np.arange(3); plt.bar(x-.18,y,.36,yerr=[lo,hi],capsize=3,label='All feasible'); plt.bar(x+.18,rb,.36,label='Robust subset (>1.96 SE)')
for i,v in enumerate(y): plt.text(i-.18,v+hi[i]+.1,f'{v:.1f}%',ha='center',fontsize=7)
plt.xticks(x,xs); plt.ylim(0,6.5); plt.xlabel('Shared failure-stress fraction (rho)'); plt.ylabel('RETI vs min-energy\ndisagreement (%)')
plt.legend(fontsize=7); plt.tight_layout(); plt.savefig('figs/fig3.png')
# Fig4
f=r['fig4']; plt.figure(figsize=(5.0,2.92),dpi=200)
plt.plot(f['rho'],f['nocancel'],'o-',ms=3.5,label='SMS+MMS, no cancellation')
plt.plot(f['rho'],f['cancel'],'s-',ms=3.5,label='SMS+MMS, with cancellation')
plt.plot(f['rho'],f['retry'],'^-',ms=3.5,label='Sequential retry/fallback')
plt.xlabel('Shared failure-stress fraction (rho)'); plt.ylabel('Mean modeled energy (relative units)'); plt.ylim(0,5.5)
plt.grid(alpha=.25); plt.legend(fontsize=7,loc='center right'); plt.tight_layout(); plt.savefig('figs/fig4.png')
