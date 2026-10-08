"""
reti_sim.py: simulation core for the ITNAC RETI paper (Sections V to VII).
Implements Eqs. (10)-(16) of the manuscript. All inputs are synthetic and author-chosen.
"""
import numpy as np
from scipy.stats import norm

ALPHA = 0.30   # setup fraction in Eq. (16)
# attempt slots: 0,1,2 = RCS attempts (original + up to two retries), 3 = SMS, 4 = MMS, 5 = multi-hop
RCS0, RCS1, RCS2, SMS, MMS, MH = range(6)
ACTIONS = ["RCS", "SMS", "MMS", "Retry", "MH", "Red"]   # Red = parallel SMS+MMS

def make_cfg(comp, L, tau, r=2, gamma=0.0, lag=20.0, overhead=0.15, sigma=None):
    """comp: dict name -> (p, e, m, sigma). names RCS, SMS, MMS, MH."""
    return dict(comp=comp, L=L, tau=tau, r=r, gamma=gamma, lag=lag, overhead=overhead)

def innovations(rng, N, kappa_needed=True):
    return dict(G=rng.standard_normal(N), EPS=rng.standard_normal((N, 6)),
                ETA=rng.standard_normal((N, 6)))

def _slots(cfg):
    c = cfg["comp"]
    order = [c["RCS"], c["RCS"], c["RCS"], c["SMS"], c["MMS"], c["MH"]]
    p = np.array([o[0] for o in order]); e = np.array([o[1] for o in order])
    m = np.array([o[2] for o in order]); s = np.array([o[3] for o in order])
    return p, e, m, s

def draw_attempts(cfg, inn, rho, kappa=1.0):
    p, e, m, s = _slots(cfg)
    G = inn["G"][:, None]
    Z = np.sqrt(rho) * G + np.sqrt(1 - rho) * inn["EPS"]
    W = kappa * Z + np.sqrt(max(1 - kappa ** 2, 0.0)) * inn["ETA"]
    D = Z < norm.ppf(p)[None, :]
    T = m[None, :] * np.exp(s[None, :] * W)
    return D, T, e

def _att_energy(e_k, T_k, h, mult, eq16):
    if not eq16:
        return np.full(T_k.shape, e_k)
    return e_k * mult * (ALPHA + (1 - ALPHA) * np.minimum(h / T_k, 1.0))

def run_actions(cfg, inn, rho, kappa=1.0, eq16=False, cancel=True, congest=True, pair=None):
    """Return dict action -> (succ_time, energy, launched). succ_time=inf if not delivered by L."""
    D, T, e = draw_attempts(cfg, inn, rho, kappa)
    L = cfg["L"]; N = D.shape[0]
    g = cfg["gamma"] if congest else 0.0
    mult = 1.0 + g * np.maximum(inn["G"], 0.0)
    out = {}
    def single(k):
        ok = D[:, k] & (T[:, k] <= L)
        st = np.where(ok, T[:, k], np.inf)
        h = np.minimum(T[:, k], L)
        en = _att_energy(e[k], T[:, k], h, mult, eq16)
        if cfg.get("failfrac") is not None: en = np.where(ok, en, cfg["failfrac"] * e[k] * mult)
        return st, en, np.ones(N)
    out["RCS"] = single(RCS0); out["SMS"] = single(SMS); out["MMS"] = single(MMS); out["MH"] = single(MH)
    # sequential retry/fallback
    seq = list(range(RCS0, RCS0 + cfg["r"] + 1)) + [SMS]
    t = np.zeros(N); done = np.zeros(N, bool); st = np.full(N, np.inf); En = np.zeros(N); cnt = np.zeros(N)
    for k in seq:
        act = (~done) & (t < L)
        h = np.minimum(T[:, k], L - t)
        ok = act & D[:, k] & (t + T[:, k] <= L)
        en_k = _att_energy(e[k], T[:, k], np.maximum(h, 1e-9), mult, eq16)
        if cfg.get("failfrac") is not None: en_k = np.where(ok, en_k, cfg["failfrac"] * e[k] * mult)
        En += np.where(act, en_k, 0.0)
        cnt += act
        st[ok] = (t + T[:, k])[ok]
        done |= ok
        t = np.where(act, t + T[:, k], t)
    out["Retry"] = (st, En, cnt)
    # parallel pair
    def parallel(ka, kb):
        okA = D[:, ka] & (T[:, ka] <= L); okB = D[:, kb] & (T[:, kb] <= L)
        Tw = np.minimum(np.where(okA, T[:, ka], np.inf), np.where(okB, T[:, kb], np.inf))
        win = np.isfinite(Tw)
        En = np.zeros(N)
        for k in (ka, kb):
            hk = np.minimum(T[:, k], L)
            if cancel:
                hk = np.where(win, np.minimum(T[:, k], Tw + cfg["lag"]), hk)
            en_k = _att_energy(e[k], T[:, k], hk, mult, eq16)
            if cfg.get("failfrac") is not None:
                okk = okA if k == ka else okB
                en_k = np.where(okk, en_k, cfg["failfrac"] * e[k] * mult)
            En += en_k
        return Tw, En + cfg["overhead"], np.full(N, 2.0)
    out["Red"] = parallel(SMS, MMS)
    if pair is not None:
        out["PAIR"] = parallel(*pair)
    return out

def summarize(res, L):
    s = {}
    for a, (st, En, cnt) in res.items():
        ok = np.isfinite(st)
        s[a] = dict(Q=float(ok.mean()), E=float(En.mean()),
                    p95=float(np.percentile(st[ok], 95)) if ok.any() else float("nan"),
                    launched=float(cnt.mean()), n=len(st))
    return s

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def wilson_lower_onesided(k, n, z=1.645):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h

def choose(summ, tau, policy, actions=ACTIONS, lam=None):
    feas = [a for a in actions if summ[a]["Q"] >= tau]
    if not feas: return None
    if policy == "RETI":  return max(feas, key=lambda a: summ[a]["Q"] / summ[a]["E"])
    if policy == "minE":  return min(feas, key=lambda a: summ[a]["E"])
    if policy == "maxQ":  return max(feas, key=lambda a: summ[a]["Q"])
    if policy == "W":     return max(feas, key=lambda a: 100 * summ[a]["Q"] - lam * summ[a]["E"])
    raise ValueError(policy)
