#!/usr/bin/env python3
"""
reproduce_reti.py: regenerates the synthetic experiments of the ITNAC manuscript
"Resilience-Energy Trade-offs in Carrier Messaging Fallback Paths" (Tables III to XII, Figures 1 to 4)
from the manuscript equations. All data are synthetic; all coefficients are author-chosen and listed below.
Usage: python3 reproduce_reti.py [outdir]          (about 10 to 15 minutes)
Seeds: Section V 20261009, perturbation 20261001, sweep 20261007, holdout 20261011.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reti_sim import *

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
R = {}   # all results

# ------------------------------------------------------------------ Table III / IV (deterministic)
PATHS6 = ["RCS", "SMS", "MMS", "Retry", "Multi-hop", "Redundant"]
R_BASE = {"RCS": 5.4, "SMS": 80.9, "MMS": 56.6, "Retry": 63.7, "Multi-hop": 42.4, "Redundant": 89.5}
E_BASE = {"RCS": 1.0, "SMS": 1.6, "MMS": 2.4, "Retry": 2.1, "Multi-hop": 2.8, "Redundant": 3.6}
R_OFF = [4, 1, -1, -4]; E_MUL = [0.94, 0.98, 1.02, 1.06]
CONDS = ["Normal", "Congested", "Partial", "Severe"]
T3 = {p: [(round(R_BASE[p] + R_OFF[c], 1), round(E_BASE[p] * E_MUL[c], 2)) for c in range(4)] for p in PATHS6}
# the manuscript prints Table III values rounded as below; use the printed cells for inputs
T3_PRINTED = {
 "RCS": [(9.4, .94), (6.4, .98), (4.4, 1.02), (1.4, 1.06)], "SMS": [(84.9, 1.50), (81.9, 1.57), (79.9, 1.63), (76.9, 1.70)],
 "MMS": [(60.6, 2.26), (57.6, 2.35), (55.6, 2.45), (52.6, 2.54)], "Retry": [(67.7, 1.97), (64.7, 2.06), (62.7, 2.14), (59.7, 2.23)],
 "Multi-hop": [(46.4, 2.63), (43.4, 2.74), (41.4, 2.86), (38.4, 2.97)], "Redundant": [(93.5, 3.38), (90.5, 3.53), (88.5, 3.67), (85.5, 3.82)]}
R["table3_max_abs_diff_vs_formula"] = max(abs(T3[p][c][0] - T3_PRINTED[p][c][0]) for p in PATHS6 for c in range(4))
qR = {c: {p: 100 * T3_PRINTED[p][i][0] / 100 / T3_PRINTED[p][i][1] for p in PATHS6} for i, c in enumerate(CONDS)}
R["qR_by_condition_SMS"] = {c: round(qR[c]["SMS"], 2) for c in CONDS}
Rw = {p: np.mean([T3_PRINTED[p][i][0] for i in range(4)]) for p in PATHS6}
Ew = {p: np.mean([T3_PRINTED[p][i][1] for i in range(4)]) for p in PATHS6}
R["qRw"] = {p: round(100 * (Rw[p] / 100) / Ew[p], 2) for p in PATHS6}
TAU4 = [80, 80, 84, 85]; TTD = {"RCS": 151.5, "SMS": 149.8, "MMS": 148.4, "Retry": 150.2, "Multi-hop": 300.0, "Redundant": 351.3}
R["table4"] = {}
for i, c in enumerate(CONDS):
    feas = [p for p in PATHS6 if T3_PRINTED[p][i][0] >= TAU4[i] and TTD[p] <= 400]
    R["table4"][c] = dict(tau=TAU4[i], feasible=feas,
                          choice=max(feas, key=lambda p: T3_PRINTED[p][i][0] / T3_PRINTED[p][i][1]) if feas else None)

# ------------------------------------------------------------------ Table V: legacy perturbation (3,000 each)
rng = np.random.default_rng(20261001); t5 = {}
for i, c in enumerate(CONDS):
    counts = {}; none = 0
    for _ in range(3000):
        best, bv = None, -1
        for p in PATHS6:
            r = T3_PRINTED[p][i][0] + rng.uniform(-5, 5); e = T3_PRINTED[p][i][1] * (1 + rng.uniform(-.15, .15))
            t = TTD[p] * (1 + rng.uniform(-.10, .10))
            if r >= TAU4[i] and t <= 400 and r / e > bv: best, bv = p, r / e
        if best is None: none += 1
        else: counts[best] = counts.get(best, 0) + 1
    top = max(counts, key=counts.get)
    t5[c] = dict(most_selected=top, freq=round(100 * counts[top] / 3000, 1), no_feasible=round(100 * none / 3000, 1))
R["table5"] = t5
print("Table V", t5, flush=True)

# ------------------------------------------------------------------ Section V scenarios
MS = {"RCS": (112, .35), "SMS": (136, .35), "MMS": (185, .35), "MH": (250, .40)}
def C(rcs, sms, mms, mh, L, tau, r=2):
    comp = {"RCS": (*rcs, *MS["RCS"]), "SMS": (*sms, *MS["SMS"]), "MMS": (*mms, *MS["MMS"]), "MH": (*mh, *MS["MH"])}
    return make_cfg(comp, L, tau, r=r)
SCEN = {}
for i, c in enumerate(CONDS):   # original four conditions from Table III
    SCEN[c] = C(*[(T3_PRINTED[k][i][0] / 100, T3_PRINTED[k][i][1]) for k in ("RCS", "SMS", "MMS", "Multi-hop")], 400, TAU4[i] / 100)
SCEN["Healthy RCS"] = C((.977, 1.2), (.88, 1.5), (.904, 2.4), (.50, 2.8), 400, .90)
SCEN["R/E reversal"] = C((.60, .9), (.88, 1.4), (.70, 2.2), (.60, 2.6), 400, .80)
SCEN["Marginal gain"] = C((.60, .9), (.90, 1.3), (.55, 2.0), (.45, 2.6), 400, .85)
SCEN["Short deadline"] = C((.50, 1.0), (.93, 1.4), (.75, 2.1), (.50, 2.6), 200, .88)
SCEN["High threshold"] = C((.40, 1.0), (.88, 1.4), (.75, 2.1), (.50, 2.6), 400, .97)
SCEN["Energy trade-off"] = C((.75, .9), (.86, 1.2), (.20, 2.0), (.45, 2.6), 400, .85)
SCEN["Near tie"] = C((.76, 1.2), (.81, 1.5), (.50, 2.4), (.45, 2.8), 400, .80)
R["scenario_inputs"] = {k: dict(L=v["L"], tau=v["tau"], comp={a: list(b) for a, b in v["comp"].items()}) for k, v in SCEN.items()}
RHOS = [0.0, 0.3, 0.6, 0.9]
rng5 = np.random.default_rng(20261009)
S5 = {}   # (scenario, rho) -> summary
for sc, cfg in SCEN.items():
    for rho in RHOS:
        inn = innovations(rng5, 100000)
        pair = (SMS, RCS0)
        S5[(sc, rho)] = summarize(run_actions(cfg, inn, rho, pair=pair), cfg["L"])
        S5[(sc, rho)]["_n"] = 100000

# Table VI
t6 = []
for c in ("Normal", "Partial", "Severe"):
    for rho in (0.0, 0.6, 0.9):
        s = S5[(c, rho)]; best = max(s["SMS"]["Q"], s["MMS"]["Q"])
        t6.append(dict(cond=c, rho=rho, pairQ=round(100 * s["Red"]["Q"], 2), gain=round(100 * (s["Red"]["Q"] - best), 2)))
R["table6"] = t6
# Partial rho=0.6 conservative decision
sp = S5[("Partial", 0.6)]
kq = int(round(sp["Red"]["Q"] * 100000)); R["partial_rho06"] = dict(Q=round(100 * sp["Red"]["Q"], 3), wilson_lb_1sided=round(100 * wilson_lower_onesided(kq, 100000), 3), tau=84,
    feasible_point=bool(sp["Red"]["Q"] >= .84), feasible_lb=bool(wilson_lower_onesided(kq, 100000) >= .84))
R["partial_rho09_any_feasible84"] = any(S5[("Partial", .9)][a]["Q"] >= .84 for a in ACTIONS)

# Table VII, agreement counts, regret
t7 = {}; agree = 0; infeasible = 0; disagree = []; reg5 = {"RETI": [], "minE": [], "maxQ": []}; reg20 = {"RETI": [], "minE": []}; feas_settings = 0
for sc, cfg in SCEN.items():
    tau = cfg["tau"]
    for rho in RHOS:
        s = S5[(sc, rho)]; ch = {p: choose(s, tau, p, lam=None) for p in ("RETI", "minE", "maxQ")}
        ch["W5"] = choose(s, tau, "W", lam=5); ch["W20"] = choose(s, tau, "W", lam=20)
        if rho == 0.6: t7[sc] = ch
        if ch["RETI"] is None: infeasible += 1; continue
        feas_settings += 1
        if ch["RETI"] == ch["minE"]: agree += 1
        else: disagree.append((sc, rho, ch["RETI"], ch["minE"]))
        for lam, store in ((5, reg5), (20, reg20)):
            feas = [a for a in ACTIONS if s[a]["Q"] >= tau]
            U = {a: 100 * s[a]["Q"] - lam * s[a]["E"] for a in feas}; best = max(U.values())
            for pol in store:
                store[pol].append(best - U[ch[pol]])
R["table7_rho06"] = t7
R["agreement"] = dict(settings=len(SCEN) * len(RHOS), infeasible=infeasible, feasible=feas_settings, agree=agree, disagreements=[list(map(str, d)) for d in disagree])
R["regret"] = dict(U5_RETI=round(float(np.mean(reg5["RETI"])), 3), U5_minE=round(float(np.mean(reg5["minE"])), 3),
                   U20_RETI=round(float(np.mean(reg20["RETI"])), 3), U20_minE=round(float(np.mean(reg20["minE"])), 3), n=feas_settings)
nt = S5[("Near tie", 0.6)]
R["near_tie_rho06"] = dict(RETI_choice=t7["Near tie"]["RETI"], minE_choice=t7["Near tie"]["minE"],
    retry_Q=round(100 * nt["Retry"]["Q"], 2), retry_E=round(nt["Retry"]["E"], 2), sms_Q=round(100 * nt["SMS"]["Q"], 2), sms_E=round(nt["SMS"]["E"], 2))
R["near_tie_gain_pp"] = round(100 * (nt["Retry"]["Q"] - nt["SMS"]["Q"]), 1)
# Table VIII
t8 = []
for state, sc in (("Healthy", "Healthy RCS"), ("Partial", "Partial")):
    s = S5[(sc, 0.6)]
    t8.append(dict(state=state, pair="SMS+MMS", Q=round(100 * s["Red"]["Q"], 2), E=round(s["Red"]["E"], 2), p95=round(s["Red"]["p95"], 1)))
    t8.append(dict(state=state, pair="SMS+RCS", Q=round(100 * s["PAIR"]["Q"], 2), E=round(s["PAIR"]["E"], 2), p95=round(s["PAIR"]["p95"], 1)))
R["table8"] = t8
print("Section V done", flush=True)

# Fig 1 data: gain vs rho
grid = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]; fig1 = {}
for c in ("Normal", "Partial", "Severe"):
    fig1[c] = []
    for rho in grid:
        s = summarize(run_actions(SCEN[c], innovations(rng5, 100000), rho), 400)
        fig1[c].append(100 * (s["Red"]["Q"] - max(s["SMS"]["Q"], s["MMS"]["Q"])))
R["fig1"] = dict(rho=grid, gain=fig1)
R["fig2"] = {a: dict(Q=100 * nt[a]["Q"], E=nt[a]["E"], p95=nt[a]["p95"]) for a in ACTIONS}

# ------------------------------------------------------------------ Section VII sweep
RNG = dict(p=dict(RCS=(.62, .99), SMS=(.70, .99), MMS=(.55, .98), MH=(.50, .96)),
           e=dict(RCS=(.80, 1.45), SMS=(1.05, 2.20), MMS=(1.60, 3.25), MH=(2.00, 3.70)),
           m=dict(RCS=(85, 185), SMS=(110, 225), MMS=(135, 280), MH=(195, 390)))
def draw_cfg(rng, copula=False):
    comp = {}
    for k in ("RCS", "SMS", "MMS", "MH"):
        if copula:
            Rm = np.array([[1, .75, -.55], [.75, 1, -.4125], [-.55, -.4125, 1]])
            u = norm.cdf(rng.multivariate_normal(np.zeros(3), Rm))
        else:
            u = rng.uniform(size=3)
        comp[k] = [RNG["p"][k][0] + u[0] * (RNG["p"][k][1] - RNG["p"][k][0]),
                   RNG["e"][k][0] + u[1] * (RNG["e"][k][1] - RNG["e"][k][0]),
                   RNG["m"][k][0] + u[2] * (RNG["m"][k][1] - RNG["m"][k][0])]
    sig = rng.uniform(.27, .55)
    comp = {k: (v[0], v[1], v[2], sig) for k, v in comp.items()}
    cfg = make_cfg(comp, L=rng.uniform(260, 540), tau=rng.uniform(.73, .92), r=int(rng.integers(0, 3)),
                   gamma=rng.uniform(.05, .65), lag=rng.uniform(8, 55), overhead=rng.uniform(.07, .28))
    return cfg
SRHO = [0.0, 0.5, 0.85]; NCFG = 1200; NEV = 4096
rngS = np.random.default_rng(20261007)
cfgs = []; inns = []
for _ in range(NCFG):
    cfgs.append(draw_cfg(rngS)); inns.append(innovations(rngS, NEV))
def eval_all(cfgs, inns, rho, **kw):
    return [summarize(run_actions(c, i, rho, eq16=True, **kw), c["L"]) for c, i in zip(cfgs, inns)]
def decisions(sums, cfgs):
    out = []
    for s, c in zip(sums, cfgs):
        out.append((choose(s, c["tau"], "RETI"), choose(s, c["tau"], "minE")))
    return out
SW = {}; t0 = time.time()
for rho in SRHO + [0.25, 0.75]:
    SW[("base", rho)] = eval_all(cfgs, inns, rho)
    SW[("nocancel", rho)] = eval_all(cfgs, inns, rho, cancel=False)
    SW[("nocong", rho)] = eval_all(cfgs, inns, rho, congest=False)
    print("sweep rho", rho, round(time.time() - t0), "s", flush=True)
t9 = []; dec = {}
for rho in SRHO:
    d = decisions(SW[("base", rho)], cfgs); dec[rho] = d
    feas = [i for i, (a, b) in enumerate(d) if a is not None]; dif = [i for i in feas if d[i][0] != d[i][1]]
    lo, hi = wilson(len(dif), len(feas))
    gain = np.mean([100 * (SW[("base", rho)][i][d[i][0]]["Q"] - SW[("base", rho)][i][d[i][1]]["Q"]) for i in dif]) if dif else float("nan")
    prem = np.mean([100 * (SW[("base", rho)][i][d[i][0]]["E"] / SW[("base", rho)][i][d[i][1]]["E"] - 1) for i in dif]) if dif else float("nan")
    # robust subset
    rob = []
    for i in feas:
        s = SW[("base", rho)][i]; tau = cfgs[i]["tau"]
        if all(abs(s[a]["Q"] - tau) > 1.96 * np.sqrt(max(s[a]["Q"] * (1 - s[a]["Q"]), 1e-9) / NEV) for a in ACTIONS): rob.append(i)
    rd = [i for i in rob if d[i][0] != d[i][1]]
    t9.append(dict(rho=rho, feasible=len(feas), diff=len(dif), pct=round(100 * len(dif) / len(feas), 2), ci=[round(100 * lo, 2), round(100 * hi, 2)],
                   gain_pp=round(float(gain), 2), premium_pct=round(float(prem), 2), robust_pct=round(100 * len(rd) / max(len(rob), 1), 2), robust_n=len(rob)))
R["table9"] = t9
# Table X and ablations
t10 = []
for rho in SRHO:
    nc = np.mean([s["Red"]["E"] for s in SW[("nocancel", rho)]]); cn = np.mean([s["Red"]["E"] for s in SW[("base", rho)]])
    rc = np.mean([s["Retry"]["launched"] for s in SW[("base", rho)]])
    t10.append(dict(rho=rho, nocancel=round(float(nc), 3), cancel=round(float(cn), 3), saving=round(float(nc - cn), 3), retry_count=round(float(rc), 3)))
R["table10"] = t10
R["congestion_adds"] = [round(float(np.mean([a["Red"]["E"] - b["Red"]["E"] for a, b in zip(SW[("base", rho)], SW[("nocong", rho)])])), 3) for rho in SRHO]
R["congestion_ablation_changes_RETI"] = []; R["cancel_ablation_changes_RETI"] = []
for rho in SRHO:
    b = [x[0] for x in decisions(SW[("base", rho)], cfgs)]
    n = [x[0] for x in decisions(SW[("nocong", rho)], cfgs)]; c = [x[0] for x in decisions(SW[("nocancel", rho)], cfgs)]
    R["congestion_ablation_changes_RETI"].append(int(sum(1 for u, v in zip(b, n) if u != v)))
    R["cancel_ablation_changes_RETI"].append(int(sum(1 for u, v in zip(b, c) if u != v)))
# naive fixed-sequence energy for Retry (charge every potential leg)
R["naive_retry_E_over_actual"] = round(float(np.mean([(c["r"] + 1) * c["comp"]["RCS"][1] + c["comp"]["SMS"][1] for c in cfgs]) /
                                         np.mean([s["Retry"]["E"] for s in SW[("base", 0.5)]])), 2)
# Fig 4 data
grid4 = [0.0, 0.25, 0.5, 0.75, 0.85]
R["fig4"] = dict(rho=grid4, nocancel=[float(np.mean([s["Red"]["E"] for s in SW[("nocancel", r)]])) for r in grid4],
                 cancel=[float(np.mean([s["Red"]["E"] for s in SW[("base", r)]])) for r in grid4],
                 retry=[float(np.mean([s["Retry"]["E"] for s in SW[("base", r)]])) for r in grid4])
# 8192-event precision check
rngP = np.random.default_rng(20261008)
inns8 = [innovations(rngP, 8192) for _ in range(NCFG)]
R["mc8192"] = []
for rho in SRHO:
    d8 = decisions(eval_all(cfgs, inns8, rho), cfgs)
    f8 = [x for x in d8 if x[0] is not None]; R["mc8192"].append(round(100 * sum(1 for a, b in f8 if a != b) / len(f8), 2))
# Table XI tertiles
taus = np.array([c["tau"] for c in cfgs]); q1, q2 = np.quantile(taus, [1 / 3, 2 / 3])
bounds = [(taus.min(), q1), (q1, q2), (q2, taus.max())]
t11 = []
for name, (lo_, hi_) in zip(("Lower", "Middle", "Upper"), bounds):
    idx = [i for i in range(NCFG) if lo_ <= taus[i] <= hi_ and (name == "Upper" or taus[i] < hi_ or name != "Lower") ]
    idx = [i for i in range(NCFG) if (taus[i] >= lo_ if name == "Lower" else taus[i] > lo_) and taus[i] <= hi_]
    row = dict(tier=name, lo=round(100 * float(lo_), 1), hi=round(100 * float(hi_), 1))
    for rho in (0.5, 0.85):
        d = dec[rho]; fe = [i for i in idx if d[i][0] is not None]; df = [i for i in fe if d[i][0] != d[i][1]]
        row[f"rho{rho}"] = (len(df), len(fe), round(100 * len(df) / max(len(fe), 1), 2))
    t11.append(row)
R["table11"] = t11
print("Section VII done", round(time.time() - t0), "s", flush=True)

# ------------------------------------------------------------------ Holdout
rngH = np.random.default_rng(20261011)
NH = 600; NTR, NVA = 1024, 2048
t12 = {}; allrates = []; gains_all = []
for design in ("Uniform", "Copula"):
    cf = []; itr = []; iva = []
    for _ in range(NH):
        cf.append(draw_cfg(rngH, copula=(design == "Copula"))); itr.append(innovations(rngH, NTR)); iva.append(innovations(rngH, NVA))
    for kappa in (0.0, 0.7):
        for rho in SRHO:
            tr = [summarize(run_actions(c, i, rho, kappa=kappa, eq16=True), c["L"]) for c, i in zip(cf, itr)]
            va = [summarize(run_actions(c, i, rho, kappa=kappa, eq16=True), c["L"]) for c, i in zip(cf, iva)]
            feas = dif = vR = vM = 0; g = []; pr = []
            for c, a, b in zip(cf, tr, va):
                cr, cm = choose(a, c["tau"], "RETI"), choose(a, c["tau"], "minE")
                if cr is None: continue
                feas += 1
                vR += b[cr]["Q"] < c["tau"]; vM += b[cm]["Q"] < c["tau"]
                if cr != cm:
                    dif += 1; g.append(100 * (b[cr]["Q"] - b[cm]["Q"])); pr.append(100 * (b[cr]["E"] / b[cm]["E"] - 1))
            t12[f"{design}|{kappa}|{rho}"] = dict(feasible=feas, diff=dif, pct=round(100 * dif / feas, 2), violR=int(vR), violM=int(vM),
                                                gain=round(float(np.mean(g)), 2) if g else None, prem=round(float(np.mean(pr)), 2) if pr else None)
            allrates.append(100 * dif / feas)
            print("holdout", design, kappa, rho, t12[f"{design}|{kappa}|{rho}"], flush=True)
R["table12_all"] = t12
R["holdout_rate_range"] = [round(min(allrates), 2), round(max(allrates), 2)]
json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
print("ALL DONE")
