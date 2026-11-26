"""P1 F2: mAP for every sampling draw (prereg section 3). One job per (det, op): loads the fast npz once,
evaluates every draw of the requested phase. Output WORK/p1/maps/<phase>/<det>/<op>.npz (key = scheme|n, values mAP x100).
Phase "A"   : A full set, bootstrap |A| on A (1000, seed +41)      -> used for predictions (p1_05_predict.py)
Phase "rest": full-set subsampling (seed +20), full-set bootstrap (+30), B partitions (+43), B bootstrap (+44), B full;
              rtdetr_l subset R: subsampling (+25) and bootstrap (+35) (secondary, not in H1-H4).
Each n uses its own stream default_rng([seed, n]).
Usage: python p1_04_draws.py <phase> <nproc> [det ...]"""
import sys, time
import numpy as np
from multiprocessing import Pool
from p1_common import *
from p1_fastmap import build, mapw_fast

NB = 1000; NPART = 200
NS_SUB = [50, 100, 166, 250, 500, 1000, 2500]; NS_BOOT = [100, 166, 250, 500, 1000, 2500]
NS_B = [50, 100, 166, 250, 500, 1000]
NS_RT_SUB = [50, 100, 166, 250, 500, 1000]; NS_RT_BOOT = [100, 166, 250, 500, 1000]


def draws(phase, det):
    """yields (key, list of weight-index specs); a spec is (positions, counts) over the detector's image set."""
    s = selection()
    if det == "rtdetr_l":
        if phase == "A": return {}
        N = len(s["rtdetr_subset"]); out = {"full|%d" % N: [(np.arange(N), None)]}
        for n in NS_RT_SUB:
            r = np.random.default_rng([SEED_P1 + 25, n]); out[f"sub|{n}"] = [(np.sort(r.choice(N, n, replace=False)), None) for _ in range(NB)]
        for n in NS_RT_BOOT + [N]:
            r = np.random.default_rng([SEED_P1 + 35, n]); out[f"boot|{n}"] = [(r.integers(0, N, n), "boot") for _ in range(NB)]
        return out
    ids = s["ids"]; N = len(ids); pos = {i: k for k, i in enumerate(ids)}
    A = np.array([pos[i] for i in s["A"]]); B = np.array([pos[i] for i in s["B"]])
    out = {}
    if phase == "A":
        out["Afull"] = [(A, None)]
        r = np.random.default_rng([SEED_P1 + 41, len(A)]); out[f"Aboot|{len(A)}"] = [(A[r.integers(0, len(A), len(A))], "boot") for _ in range(NB)]
        return out
    out["full"] = [(np.arange(N), None)]; out["Bfull"] = [(B, None)]
    for n in NS_SUB:
        r = np.random.default_rng([SEED_P1 + 20, n]); out[f"sub|{n}"] = [(np.sort(r.choice(N, n, replace=False)), None) for _ in range(NB)]
    for n in NS_BOOT + [N]:
        r = np.random.default_rng([SEED_P1 + 30, n]); out[f"boot|{n}"] = [(r.integers(0, N, n), "boot") for _ in range(NB)]
    for n in NS_B:
        r = np.random.default_rng([SEED_P1 + 43, n]); k = len(B) // n; lst = []
        for _ in range(NPART):
            p = B[r.permutation(len(B))]
            lst += [(np.sort(p[j * n:(j + 1) * n]), None) for j in range(k)]
        out[f"Bpart|{n}"] = lst
        r = np.random.default_rng([SEED_P1 + 44, n]); out[f"Bboot|{n}"] = [(B[r.integers(0, len(B), n)], "boot") for _ in range(NB)]
    return out


def weights(spec, N):
    p, kind = spec
    if kind == "boot": return np.bincount(p, minlength=N).astype(np.float64)
    w = np.zeros(N); w[p] = 1.0; return w


def job(a):
    phase, det, op = a
    fo = P1 / "maps" / phase / det / f"{op}.npz"
    if fo.exists(): return (det, op, "skip")
    t = time.time(); s = selection(); tag = "rt" if det == "rtdetr_l" else "full"
    ids = s["rtdetr_subset"] if det == "rtdetr_l" else s["ids"]; N = len(ids)
    z = build(det, op, ids, tag); D = draws(phase, det); out = {}
    for k, specs in D.items():
        out[k] = np.array([100 * mapw_fast(z, weights(sp, N)) for sp in specs])
    fo.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(fo, **out)
    return (det, op, round(time.time() - t))


def jobs_for(phase, dets):
    J = []
    for det, op, ids in det_plan():
        if dets and det not in dets: continue
        if phase == "A" and det == "rtdetr_l": continue
        J.append((phase, det, op))
    return J


if __name__ == "__main__":
    phase, nproc = sys.argv[1], int(sys.argv[2]); dets = sys.argv[3:]
    J = jobs_for(phase, dets); t0 = time.time(); print("jobs", len(J), flush=True)
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, J, chunksize=1): print(r, round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)
