#!/usr/bin/env python3
"""
extend_reti.py: reviewer-requested extension. Re-runs the Section VII sweep and the holdout with energies drawn from the
published-measurement calibration (Section VIII) instead of uniform ranges, adds a conservative (Wilson lower bound) screen,
paired confidence intervals, and weighted/Pareto-knee comparison policies.
Usage: python3 extend_reti.py [outdir]   (reads code/analysis/calibrated_energy.py for the cost model)
Seeds: sweep configs 20261007 (same as reproduce_reti.py), holdout 20261011, calibrated energy draws 20261021.
"""
import json, os, sys, time
import numpy as np
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reti_sim import *
CAL = os.environ.get("RETI_CAL_DIR", os.path.join(HERE, "..", "analysis"))
sys.path.insert(0, CAL); sys.path.insert(0, os.path.join(CAL, "..", "data"))
import calibrated_energy as ce
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "output"); os.makedirs(OUT, exist_ok=True)

RNG = dict(p=dict(RCS=(.62, .99), SMS=(.70, .99), MMS=(.55, .98), MH=(.50, .96)),
           e=dict(RCS=(.80, 1.45), SMS=(1.05, 2.20), MMS=(1.60, 3.25), MH=(2.00, 3.70)),
           m=dict(RCS=(85, 185), SMS=(110, 225), MMS=(135, 280), MH=(195, 390)))
def draw_cfg(rng, copula=False):      # identical to reproduce_reti.py
    comp = {}
    for k in ("RCS", "SMS", "MMS", "MH"):
        if copula:
            Rm = np.array([[1, .75, -.55], [.75, 1, -.4125], [-.55, -.4125, 1]]); u = norm.cdf(rng.multivariate_normal(np.zeros(3), Rm))
        else: u = rng.uniform(size=3)
        comp[k] = [RNG["p"][k][0] + u[0] * (RNG["p"][k][1] - RNG["p"][k][0]), RNG["e"][k][0] + u[1] * (RNG["e"][k][1] - RNG["e"][k][0]),
                   RNG["m"][k][0] + u[2] * (RNG["m"][k][1] - RNG["m"][k][0])]
    sig = rng.uniform(.27, .55); comp = {k: (v[0], v[1], v[2], sig) for k, v in comp.items()}
    return make_cfg(comp, L=rng.uniform(260, 540), tau=rng.uniform(.73, .92), r=int(rng.integers(0, 3)), gamma=rng.uniform(.05, .65),
                    lag=rng.uniform(8, 55), overhead=rng.uniform(.07, .28))

def cal_energy(rng):
    """One draw of the calibration assumptions (ranges as in calibrated_energy.py); returns per-attempt cost relative to a successful RCS."""
    c = ce.path_costs(V=rng.uniform(3.6, 4.35), tail_s=rng.uniform(*ce.TAIL_S), tail_frac=rng.uniform(.2, 1), amort=rng.uniform(0, 1),
                      sms_frac=rng.uniform(.25, 1), mms_kb=rng.uniform(50, 300), sig=rng.choice(["good", "medium", "poor"]), payload_kb=rng.uniform(.5, 5))
    rcs = c["RCS Direct Delivery"]; V = None
    sms = c["SMS Fallback (SS7)"] - ce.FAIL; mms = c["MMS Fallback"] - ce.FAIL
    return dict(RCS=1.0, SMS=sms / rcs, MMS=mms / rcs, MH=c["Multi-Hop Interworking"] / rcs), ce.FAIL / rcs

def with_cal(cfg, ecal, ff):
    comp = {k: (v[0], ecal[k], v[2], v[3]) for k, v in cfg["comp"].items()}
    c2 = dict(cfg); c2["comp"] = comp; c2["failfrac"] = ff; return c2

def knee(summ, tau, actions=ACTIONS):
    feas = [a for a in actions if summ[a]["Q"] >= tau]
    if not feas: return None
    nd = [a for a in feas if not any((summ[b]["Q"] >= summ[a]["Q"] and summ[b]["E"] <= summ[a]["E"] and (summ[b]["Q"] > summ[a]["Q"] or summ[b]["E"] < summ[a]["E"])) for b in feas)]
    qs = [summ[a]["Q"] for a in nd]; es = [summ[a]["E"] for a in nd]
    qr = max(qs) - min(qs) + 1e-9; er = max(es) - min(es) + 1e-9
    return min(nd, key=lambda a: ((max(qs) - summ[a]["Q"]) / qr) ** 2 + ((summ[a]["E"] - min(es)) / er) ** 2)

def choose_cons(summ, tau, policy, n, lam=None):
    """same as choose but feasibility = one-sided 95% Wilson lower bound on Q >= tau"""
    feas = [a for a in ACTIONS if wilson_lower_onesided(int(round(summ[a]["Q"] * n)), n) >= tau]
    if not feas: return None
    if policy == "RETI": return max(feas, key=lambda a: summ[a]["Q"] / summ[a]["E"])
    if policy == "minE": return min(feas, key=lambda a: summ[a]["E"])

def pol(summ, tau, name):
    if name == "RETI": return choose(summ, tau, "RETI")
    if name == "minE": return choose(summ, tau, "minE")
    if name == "maxQ": return choose(summ, tau, "maxQ")
    if name == "W5": return choose(summ, tau, "W", lam=5)
    if name == "W20": return choose(summ, tau, "W", lam=20)
    if name == "Knee": return knee(summ, tau)
POLS = ["RETI", "minE", "maxQ", "W5", "W20", "Knee"]
def ci(x, B=2000):
    """percentile bootstrap 95% interval for the mean of the paired differences (resampling disagreement configurations)"""
    x = np.asarray(x, float)
    if len(x) < 2: return [None, None]
    rb = np.random.default_rng(99); m = rb.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]

def scale_cfg(c, s, sL=1.0):
    """s multiplies all times (medians, deadline, cancellation lag); sL multiplies the deadline only"""
    if s == 1 and sL == 1: return c
    c2 = dict(c); c2["comp"] = {k: (v[0], v[1], v[2] * s, v[3]) for k, v in c["comp"].items()}
    c2["L"] = c["L"] * s * sL; c2["lag"] = c["lag"] * s; return c2

def run_sweep(seed, scale=1.0, scaleL=1.0, eseed=20261021, rhos=(0.0, 0.5, 0.85), nev=4096, ncfg=1200, detail=False):
    rngS = np.random.default_rng(seed); rngE = np.random.default_rng(eseed)
    cfgs = []; inns = []; ccfgs = []
    for _ in range(ncfg):
        c = draw_cfg(rngS); inns.append(innovations(rngS, nev)); cfgs.append(scale_cfg(c, scale, scaleL))
        ec, ff = cal_energy(rngE); ccfgs.append(with_cal(cfgs[-1], ec, ff))
    out = {}
    for rho in rhos:
        row = {}; dec = {}
        for name, cs in (("uniform", cfgs), ("calibrated", ccfgs)):
            S = [summarize(run_actions(c, i, rho, eq16=True), c["L"]) for c, i in zip(cs, inns)]
            d = [(pol(s, c["tau"], "RETI"), pol(s, c["tau"], "minE")) for s, c in zip(S, cs)]; dec[name] = d
            feas = [i for i, x in enumerate(d) if x[0] is not None]; dif = [i for i in feas if d[i][0] != d[i][1]]
            lo, hi = wilson(len(dif), len(feas))
            row[name] = dict(feasible=len(feas), diff=len(dif), pct=round(100 * len(dif) / max(len(feas), 1), 2), ci=[round(100 * lo, 2), round(100 * hi, 2)],
                gain=round(float(np.mean([100 * (S[i][d[i][0]]["Q"] - S[i][d[i][1]]["Q"]) for i in dif])), 2) if dif else None,
                prem=round(float(np.mean([100 * (S[i][d[i][0]]["E"] / S[i][d[i][1]]["E"] - 1) for i in dif])), 2) if dif else None)
            if detail:
                row[name]["infeasible_pct"] = round(100 * (1 - len(feas) / ncfg), 2)
                row[name]["reti_choice_share"] = {a: round(100 * sum(1 for i in feas if d[i][0] == a) / len(feas), 1) for a in ACTIONS}
                Ls = np.array([c["L"] for c in cs]); q1, q2 = np.quantile(Ls, [1 / 3, 2 / 3]); reg = {}
                for nm, m in (("short", Ls <= q1), ("mid", (Ls > q1) & (Ls <= q2)), ("long", Ls > q2)):
                    f = [i for i in feas if m[i]]; dd = [i for i in f if d[i][0] != d[i][1]]; reg[nm] = [len(dd), len(f), round(100 * len(dd) / max(len(f), 1), 2)]
                row[name]["by_deadline"] = reg; row[name]["L_bounds"] = [round(float(Ls.min())), round(float(q1)), round(float(q2)), round(float(Ls.max()))]
        row["reti_choice_changed"] = sum(1 for a, b in zip(dec["uniform"], dec["calibrated"]) if a[0] != b[0])
        out[str(rho)] = row
    return out, np.array([c["comp"]["SMS"][1] / c["comp"]["RCS"][1] for c in cfgs]), np.array([cal_energy(np.random.default_rng(eseed))[0]["SMS"]])

def run_holdout(seed, scale=1.0, eseed=20261022, nh=600, ntr=1024, nva=2048, kappas=(0.0, 0.7), rhos=(0.0, 0.5, 0.85)):
    rngH = np.random.default_rng(seed); rngE2 = np.random.default_rng(eseed); H = {}
    for design in ("Uniform", "Copula"):
        cf = []; itr = []; iva = []; ccf = []
        for _ in range(nh):
            c = scale_cfg(draw_cfg(rngH, copula=(design == "Copula")), scale); itr.append(innovations(rngH, ntr)); iva.append(innovations(rngH, nva)); cf.append(c)
            ec, ff = cal_energy(rngE2); ccf.append(with_cal(c, ec, ff))
        for kappa in kappas:
            for rho in rhos:
                for en, cs in (("uniform", cf), ("calibrated", ccf)):
                    tr = [summarize(run_actions(c, i, rho, kappa=kappa, eq16=True), c["L"]) for c, i in zip(cs, itr)]
                    va = [summarize(run_actions(c, i, rho, kappa=kappa, eq16=True), c["L"]) for c, i in zip(cs, iva)]
                    n = 0; g = []; pr = []; cf_n = 0; cvR = cvM = 0; comR = comM = 0
                    P = {p: dict(Q=[], E=[], v=0, dis=0) for p in POLS}
                    for c, a, b in zip(cs, tr, va):
                        ch = {p: pol(a, c["tau"], p) for p in POLS}
                        if ch["RETI"] is None: continue
                        n += 1
                        for p in POLS:
                            P[p]["Q"].append(100 * b[ch[p]]["Q"]); P[p]["E"].append(b[ch[p]]["E"]); P[p]["v"] += b[ch[p]]["Q"] < c["tau"]; P[p]["dis"] += ch[p] != ch["RETI"]
                        if ch["RETI"] != ch["minE"]:
                            g.append(100 * (b[ch["RETI"]]["Q"] - b[ch["minE"]]["Q"])); pr.append(100 * (b[ch["RETI"]]["E"] / b[ch["minE"]]["E"] - 1))
                        cr, cm = choose_cons(a, c["tau"], "RETI", ntr), choose_cons(a, c["tau"], "minE", ntr)
                        if cr is not None:
                            cf_n += 1; cvR += b[cr]["Q"] < c["tau"]; cvM += b[cm]["Q"] < c["tau"]
                            comR += b[ch["RETI"]]["Q"] < c["tau"]; comM += b[ch["minE"]]["Q"] < c["tau"]   # point-estimate choices on the common set
                    H[f"{design}|{kappa}|{rho}|{en}"] = dict(feasible=n, diff=len(g), pct=round(100 * len(g) / n, 2), gain=round(float(np.mean(g)), 2) if g else None, gain_ci=ci(g),
                        prem=round(float(np.mean(pr)), 2) if pr else None, prem_ci=ci(pr), violR=int(P["RETI"]["v"]), violM=int(P["minE"]["v"]),
                        cons_feasible=cf_n, cons_violR=int(cvR), cons_violM=int(cvM), common_violR=int(comR), common_violM=int(comM),
                        policies={p: dict(Q=round(float(np.mean(P[p]["Q"])), 2), E=round(float(np.mean(P[p]["E"])), 3), viol=int(P[p]["v"]), dis_vs_RETI=int(P[p]["dis"])) for p in POLS})
    return H

R = {}; t0 = time.time()
# A. main sweep (seed 20261007) with detail
A, sweep_ratio_u, _ = run_sweep(20261007, detail=True)
rngE = np.random.default_rng(20261021); ratios = np.array([cal_energy(rngE)[0]["SMS"] for _ in range(1200)])
R["sms_rcs_ratio"] = dict(cal_median=float(np.median(ratios)), cal_p5_p95=[float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))],
        sweep_min=float(sweep_ratio_u.min()), share_cal_below_sweep_min=float((ratios < sweep_ratio_u.min()).mean()))
R["sweep_cal"] = A; print("A", round(time.time() - t0), "s", flush=True)
# B. main holdout (seed 20261011)
R["holdout"] = run_holdout(20261011); print("B", round(time.time() - t0), "s", flush=True)
# C. seed replication: 5 independent configuration seeds for the sweep, 3 for the holdout
R["seed_sweep"] = {}
for sd in (20261101, 20261102, 20261103, 20261104, 20261105):
    R["seed_sweep"][str(sd)], _, _ = run_sweep(sd, eseed=sd + 1000)
print("C1", round(time.time() - t0), "s", flush=True)
R["seed_holdout"] = {}
for sd in (20261201, 20261202, 20261203):
    R["seed_holdout"][str(sd)] = run_holdout(sd, eseed=sd + 1000)
print("C2", round(time.time() - t0), "s", flush=True)
# D. time-scale checks. Multiplying every time by 10 (second-scale medians, deadlines, lag) must leave results unchanged because the model depends
# only on ratios; the informative test changes the deadline-to-latency ratio.
R["scale10_sweep"], _, _ = run_sweep(20261007, scale=10.0)
R["deadline_ratio_sweep"] = {}
for m in (0.5, 2.0, 4.0):
    R["deadline_ratio_sweep"][str(m)], _, _ = run_sweep(20261007, scaleL=m)
print("D", round(time.time() - t0), "s", flush=True)
json.dump(R, open(os.path.join(OUT, "results_extension.json"), "w"), indent=1, default=str)
print("ALL DONE")
