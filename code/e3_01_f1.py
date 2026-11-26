"""S2FIX F1 (post hoc): bias of BD at size n and coverage of the percentile bootstrap CI (yolo26n, iso L8, mAP).
Definitions S2/E_S2FIX_H44.md 0/F1. Checkpoints WORK/p1/e3/f1b_<ds>_<n>.npz (resumable, time caps).
Usage: python e3_01_f1.py a | b | c"""
import sys, time
import numpy as np
from e3_common import *
from e3_fastboot import batch_map_wap

E3 = P1 / "e3"; NS_A = [50, 100, 166, 500, 1000, 2500]
CAPS = [("voc", 166, 0.5), ("coco", 166, 0.9), ("voc", 500, 0.9), ("coco", 500, 1.2)]
PAIRS_B = [("jpeg", "webp"), ("webp", "avif")]; DS_ID = {"coco": 0, "voc": 1}; S_MAX, B_IN = 300, 300


def cols(c): return [OPS_CORE.index(f"{c}_{l}") for l in range(8)]


def bd_rows(R, Q, a, b):
    """R: (B, 24) bpp, Q: (B, 24) mAP -> BD iso per row"""
    ca, cb = cols(a), cols(b); return np.array([bd_iso(R[j, ca], Q[j, ca], R[j, cb], Q[j, cb]) for j in range(R.shape[0])])


def full_bd(ds, Bm, Z):
    N = Bm.shape[0]; R = Bm.mean(0)[None]; Q = np.array([[100 * batch_map_wap(Z[o], np.arange(N), np.ones((1, N)), THREADS)[0, 0] for o in OPS_CORE]])
    return {f"{a}->{b}": float(bd_rows(R, Q, a, b)[0]) for a, b in PAIRS}


def part_a():
    out = {"label": "POST HOC F1a (bias), yolo26n, iso L8, mAP", "cells": {}}
    for ds in ("coco", "voc"):
        Bm = bpp_matrix(ds, OPS_CORE); N = Bm.shape[0]; Z = {o: fast(ds, "yolo26n", o) for o in OPS_CORE}
        F = full_bd(ds, Bm, Z); out[f"bd_full|{ds}"] = F
        if ds == "coco":
            from p1_04_draws import draws
            D = draws("rest", "yolo26n"); M = {o: np.load(P1 / "maps" / "rest" / "yolo26n" / f"{o}.npz") for o in OPS_CORE}
        for n in NS_A:
            t = time.time()
            if ds == "coco":
                specs = [p for p, _ in D[f"sub|{n}"]]; Q = np.stack([M[o][f"sub|{n}"] for o in OPS_CORE], 1)
            else:
                r = np.random.default_rng([SEED_F + 300, n]); specs = [np.sort(r.choice(N, n, replace=False)) for _ in range(1000)]
                Q = np.stack([maps_spec(ds, "yolo26n", o, specs, z=Z[o])[:, 0] for o in OPS_CORE], 1)
            R = np.array([Bm[p].mean(0) for p in specs])
            for a, b in PAIRS:
                v = bd_rows(R, Q, a, b); v = v[~np.isnan(v)]; d = v - F[f"{a}->{b}"]
                rb = np.random.default_rng(SEED_F + 301); med_bs = [np.median(rb.choice(d, d.size)) for _ in range(1000)]
                out["cells"][f"{ds}|{a}->{b}|{n}"] = {"n_valid": int(v.size), "bias_mean": float(d.mean()), "mc_se_mean": float(d.std(ddof=1) / np.sqrt(d.size)),
                                                     "bias_median": float(np.median(d)), "mc_se_median": float(np.std(med_bs, ddof=1)), "sd": float(d.std(ddof=1))}
            print(ds, n, round(time.time() - t), {k.split("|")[1]: round(v["bias_mean"], 2) for k, v in out["cells"].items() if k.startswith(ds) and k.endswith(f"|{n}")}, flush=True)
    save_json(out, RES / "e3_f1a.json")


def part_b():
    E3.mkdir(parents=True, exist_ok=True); A = load_json(RES / "e3_f1a.json")
    for ds, n, cap in CAPS:
        f = E3 / f"f1b_{ds}_{n}.npz"; done = dict(np.load(f)) if f.exists() else {"lo": np.zeros((0, 2)), "hi": np.zeros((0, 2)), "pt": np.zeros((0, 2))}
        s0 = done["lo"].shape[0]
        if s0 >= S_MAX: continue
        Bm = bpp_matrix(ds, OPS_CORE); N = Bm.shape[0]; Z = {o: fast(ds, "yolo26n", o) for o in OPS_CORE}; t0 = time.time()
        ro = np.random.default_rng([SEED_F + 302, n, DS_ID[ds]]); outer = [np.sort(ro.choice(N, n, replace=False)) for _ in range(S_MAX)]
        lo, hi, pt = list(done["lo"]), list(done["hi"]), list(done["pt"])
        for s in range(s0, S_MAX):
            if (time.time() - t0) / 3600 > cap: break
            smp = outer[s]; ri = np.random.default_rng([SEED_F + 303, n, DS_ID[ds], s])
            W = np.vstack([np.ones(n)] + [np.bincount(ri.integers(0, n, n), minlength=n) for _ in range(B_IN)]).astype(float)
            Q = np.stack([100 * batch_map_wap(Z[o], smp, W, THREADS)[:, 0] for o in OPS_CORE], 1); R = (W @ Bm[smp]) / n
            row_l, row_h, row_p = [], [], []
            for a, b in PAIRS_B:
                v = bd_rows(R, Q, a, b); c = ci(v[1:]); row_l.append(c[0] if c else np.nan); row_h.append(c[1] if c else np.nan); row_p.append(v[0])
            lo.append(row_l); hi.append(row_h); pt.append(row_p)
            np.savez(f, lo=np.array(lo), hi=np.array(hi), pt=np.array(pt))
            if s % 10 == 0: print(ds, n, s, round(time.time() - t0), flush=True)
        print(ds, n, "outer done", len(lo), round(time.time() - t0), flush=True)


def part_c():
    A = load_json(RES / "e3_f1a.json"); out = {"label": "POST HOC F1b/c (coverage), yolo26n, iso L8, mAP; inner bootstrap 300", "cells": {}}
    for ds, n, cap in CAPS:
        f = E3 / f"f1b_{ds}_{n}.npz"
        if not f.exists(): continue
        z = np.load(f)
        for j, (a, b) in enumerate(PAIRS_B):
            T = A[f"bd_full|{ds}"][f"{a}->{b}"]; lo, hi = z["lo"][:, j], z["hi"][:, j]; ok = ~np.isnan(lo); S = int(ok.sum())
            k = int(np.sum((lo[ok] <= T) & (T <= hi[ok]))); bias = A["cells"][f"{ds}|{a}->{b}|{n}"]["bias_mean"]
            kc = int(np.sum((lo[ok] - bias <= T) & (T <= hi[ok] - bias)))
            E = {"S_outer": S, "coverage": k / S, "wilson95": wilson(k, S), "miss_low": int(np.sum(hi[ok] < T)), "miss_high": int(np.sum(lo[ok] > T)),
                 "mean_halfwidth": float(np.mean((hi[ok] - lo[ok]) / 2)), "true_bd_full": T, "mean_point": float(np.nanmean(z["pt"][:, j]))}
            if k / S < 0.90: E.update({"bias_used": bias, "coverage_bias_corrected": kc / S, "wilson95_bc": wilson(kc, S)})
            out["cells"][f"{ds}|{a}->{b}|{n}"] = E; print(ds, n, a, b, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in E.items()})
    save_json(out, RES / "e3_f1bc.json")


if __name__ == "__main__":
    {"a": part_a, "b": part_b, "c": part_c}[sys.argv[1]]()
