"""Optional figure: coverage versus observed synthetic validation-floor violations.
Run after run_reti_extended.py. Uses matplotlib default colors without custom styling.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
p=pd.read_csv(HERE/'results/policy_coverage_and_outcomes.csv')
pols=['RETI','minE','Wilson-minE','Robust-LCB','Certified-RETI','MC-reference']
for model in ('U','L'):
    s=p[(p.energy==model)&(p.policy.isin(pols))]
    fig,ax=plt.subplots(figsize=(7.2,4.8),dpi=160)
    for row in s.itertuples():
        ax.scatter(row.coverage_pct,row.violation_rate_given_served_pct,s=65)
        ax.annotate(row.policy,(row.coverage_pct,row.violation_rate_given_served_pct),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Policy coverage across all 2,400 synthetic test settings (%)')
    ax.set_ylabel('Held-out validation floor violations among selected cases (%)')
    ax.set_title(f'RETI policy coverage / risk trade-off — {model} energy')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(HERE/'results'/f'coverage_vs_violations_{model}.png',bbox_inches='tight')
    plt.close(fig)
    print('wrote',f'coverage_vs_violations_{model}.png')
