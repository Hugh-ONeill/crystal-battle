"""Offline study: is ANY early-game signal worth trusting?

Baseline of record: the leaf eval's opening Brier is 0.2496-0.2559, i.e. at or
worse than always-saying-50/50. This asks whether cheap observable features
(material, hazards, faints) or the pre-battle opponent prior do better, using
honest validation (leave-one-out for priors, 5-fold CV for fitted models).
"""
import glob, json, re, sys
from collections import defaultdict
import numpy as np

HP_RE = re.compile(r"^(\d+)/(\d+)")
HAZ = {"Stealth Rock": 1.0, "Spikes": 1.0, "Toxic Spikes": 1.0, "Sticky Web": 1.0}

def frac(tok):
    tok = tok.strip()
    if tok.startswith("0 fnt") or tok == "0":
        return 0.0
    m = HP_RE.match(tok)
    return int(m.group(1)) / int(m.group(2)) if m else None

def parse(path, me="PAC-Crystal"):
    games = {}
    cur = None
    for raw in open(path, errors="replace"):
        m = re.search(r">(battle-gen9oulongtimer-\d+)", raw)
        if m:
            cur = m.group(1)
            games.setdefault(cur, dict(us=None, opp=None, res=None, snap={},
                                       hp={}, haz=defaultdict(float),
                                       faint=defaultdict(int), turn=0))
        if cur is None: continue
        g = games[cur]
        i = raw.find("|")
        if i < 0: continue
        p = raw[i:].rstrip("\n").split("|")[1:]
        if not p: continue
        tag = p[0]
        try:
            if tag == "player" and len(p) > 2:
                if p[2].strip() == me: g["us"] = p[1]
                elif p[2].strip(): g["opp"] = p[2].strip()
            elif tag == "turn":
                g["turn"] = int(p[1])
                us, them = g["us"], ("p1" if g["us"] == "p2" else "p2")
                if us:
                    def team(side):
                        seen = [v for (s, _), v in g["hp"].items() if s == side]
                        return sum(seen) + (6 - len(seen))
                    g["snap"][g["turn"]] = dict(
                        hp=team(us) - team(them),
                        haz=g["haz"][them] - g["haz"][us],
                        ft=g["faint"][them] - g["faint"][us])
            elif tag in ("switch", "drag", "-damage", "-heal") and len(p) > 2:
                ident = p[1]
                if len(ident) > 4 and ident[0] == "p":
                    side, nick = ident[:2], ident.split(": ", 1)[-1]
                    f = frac(p[3] if tag in ("switch", "drag") else p[2])
                    if f is not None: g["hp"][(side, nick)] = f
            elif tag == "faint" and len(p) > 1:
                g["faint"][p[1][:2]] += 1
            elif tag in ("-sidestart", "-sideend") and len(p) > 2:
                side = p[1][:2]
                name = p[2].replace("move: ", "").strip()
                if name in HAZ:
                    g["haz"][side] += HAZ[name] * (1 if tag == "-sidestart" else -1)
            elif tag == "win" and len(p) > 1:
                g["res"] = 1.0 if p[1].strip() == me else 0.0
            elif tag == "tie":
                g["res"] = 0.5
        except Exception:
            pass
    return {k: v for k, v in games.items() if v["res"] is not None and v["us"] and v["snap"]}

def evals(stamp):
    out = {}
    try:
        for line in open(f"showdown/desk_reads_{stamp}.jsonl", errors="replace"):
            d = json.loads(line)
            out[d["battle_tag"]] = d["reads"]
    except Exception:
        pass
    return out

def brier(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    return float(np.mean((pred - y) ** 2))

def fit_logistic(X, y, l2=1.0, iters=4000, lr=0.1):
    X = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        grad = X.T @ (p - y) / len(y)
        grad[1:] += l2 * w[1:] / len(y)
        w -= lr * grad
    return w

def predict(w, X):
    X = np.column_stack([np.ones(len(X)), X])
    return 1 / (1 + np.exp(-X @ w))

def cv_brier(X, y, folds=5):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim == 1: X = X[:, None]
    idx = np.arange(len(y)); rng = np.random.default_rng(0); rng.shuffle(idx)
    preds = np.zeros(len(y))
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = fit_logistic((X[tr] - mu) / sd, y[tr])
        preds[te] = predict(w, (X[te] - mu) / sd)
    return brier(preds, y)

# ---- load ----
stamps = [re.search(r"overnight_(\d{8}_\d{6})_", p).group(1)
          for p in sorted(glob.glob("showdown/bench/overnight_*_ladder.log"))]
rows = []
for s in stamps:
    gs = parse(f"showdown/bench/overnight_{s}_ladder.log")
    ev = evals(s)
    for tag, g in gs.items():
        reads = ev.get(tag, [])
        rows.append(dict(tag=tag, opp=g["opp"] or "?", y=g["res"],
                         snap=g["snap"], reads=reads))
print(f"parsed {len(rows)} finished games with state snapshots "
      f"({sum(1 for r in rows if r['reads'])} with eval trajectories)\n")

for T in (3, 5, 9):
    sub = [r for r in rows if T in r["snap"]]
    y = np.array([r["y"] for r in sub])
    if len(sub) < 40: continue
    hp = np.array([r["snap"][T]["hp"] for r in sub])
    haz = np.array([r["snap"][T]["haz"] for r in sub])
    ft = np.array([r["snap"][T]["ft"] for r in sub])
    # leave-one-out opponent prior
    tot, win = defaultdict(int), defaultdict(float)
    for r in sub: tot[r["opp"]] += 1; win[r["opp"]] += r["y"]
    prior = np.array([(win[r["opp"]] - r["y"]) / (tot[r["opp"]] - 1)
                      if tot[r["opp"]] > 1 else float(np.mean(y)) for r in sub])
    # eval at/just before T
    def ev_at(r):
        v = [s for t, s in r["reads"] if t <= T]
        return v[-1] if v else None
    have = [i for i, r in enumerate(sub) if ev_at(r) is not None]
    ev_arr = np.array([ev_at(sub[i]) for i in have])
    base = float(np.mean(y))
    print(f"=== cutoff T{T}  (n={len(sub)} games, base rate {base:.2f}) ===")
    print(f"  {'always 0.50':28s} Brier {brier(np.full(len(y), .5), y):.4f}")
    print(f"  {'base rate (constant)':28s} Brier {brier(np.full(len(y), base), y):.4f}")
    if len(have) > 40:
        print(f"  {'LEAF EVAL (baseline)':28s} Brier {brier(ev_arr, y[have]):.4f}   n={len(have)}")
    print(f"  {'opponent prior (LOO)':28s} Brier {brier(prior, y):.4f}")
    print(f"  {'material (HP diff)':28s} Brier {cv_brier(hp, y):.4f}")
    print(f"  {'hazards':28s} Brier {cv_brier(haz, y):.4f}")
    print(f"  {'faints':28s} Brier {cv_brier(ft, y):.4f}")
    print(f"  {'material+hazards+faints':28s} Brier {cv_brier(np.column_stack([hp,haz,ft]), y):.4f}")
    print(f"  {'opp prior + material':28s} Brier {cv_brier(np.column_stack([prior,hp]), y):.4f}")
    print(f"  {'opp prior + mat + haz + ft':28s} Brier {cv_brier(np.column_stack([prior,hp,haz,ft]), y):.4f}")
    if len(have) > 40:
        pr_h, hp_h = prior[have], hp[have]
        print(f"  {'eval + opp prior':28s} Brier {cv_brier(np.column_stack([ev_arr,pr_h]), y[have]):.4f}   n={len(have)}")
        print(f"  {'eval + opp prior + material':28s} Brier {cv_brier(np.column_stack([ev_arr,pr_h,hp_h]), y[have]):.4f}   n={len(have)}")
    print()
