"""E step 1 (exploratory, COCO): BD-rate error budget from stored mAP draws (definitions: S1/E_STEP1_H44.md section 1).
No detector run, no encoding. Stored draws: WORK/p1/maps/rest/<det>/<op>.npz; bpp: WORK/p1/bpp.npz.
B1-B3 full set: BD for every (method M, domain, grid G, cell). B4: 300 first stored bootstrap draws boot|n (seed P1+30 / +35).
B5: detector differences with paired bootstrap CI (yolo26n vs yolo26s on boot|4952; rtdetr_l vs yolo on the 1,500-image subset,
yolo mAP recomputed from stored detection-GT matches with p1_fastmap.mapw_fast, cached in WORK/p1/e1).
Output results/e1_budget.json.  Usage: python e1_01_budget.py"""
import time, warnings
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from p1_common import *
from p1_bd import bd_rate, bd_iso, iso_curve, computable_std
from p1_06_stats import load, bpp_table, curve_cols
from p1_04_draws import draws

warnings.simplefilter("ignore")
EPS = 1e-3; NB = 300
GRIDS = {"L4": [0, 1, 2, 3], "L5": [0, 1, 2, 3, 4], "L6": list(range(6)), "L7": list(range(7)), "L8": list(range(8)), "S4": [0, 2, 4, 6]}
METHODS = ["poly", "pchip", "akima", "iso"]; DOMAINS = ["map", "logit"]
CORE_PAIRS = [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]
CELLS = [(a, b, d) for d in ("yolo26n", "yolo26s") for a, b in CORE_PAIRS] + [(c, "hevc", "yolo26n") for c in CORE] + \
        [(a, b, "rtdetr_l") for a, b in CORE_PAIRS]
NS = {"yolo26n": [166, 500, 1000, 2500], "yolo26s": [166, 500, 1000, 2500], "rtdetr_l": [166, 500, 1000]}
E1 = P1 / "e1"


def bd_poly(r1, q1, r2, q2):
    if len(np.unique(q1)) < 4 or len(np.unique(q2)) < 4: return np.nan
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    if hi <= lo: return np.nan
    p1 = np.polyint(np.polyfit(q1, np.log(r1), 3)); p2 = np.polyint(np.polyfit(q2, np.log(r2), 3))
    d = (np.polyval(p2, hi) - np.polyval(p2, lo)) - (np.polyval(p1, hi) - np.polyval(p1, lo))
    return float((np.exp(d / (hi - lo)) - 1) * 100)


def bd_akima(r1, q1, r2, q2):
    if not computable_std(q1, q2): return np.nan
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    f1 = Akima1DInterpolator(q1, np.log(r1)); f2 = Akima1DInterpolator(q2, np.log(r2))
    return float((np.exp((f2.integrate(lo, hi) - f1.integrate(lo, hi)) / (hi - lo)) - 1) * 100)


def bd(m, r1, q1, r2, q2):
    if m == "poly": return bd_poly(r1, q1, r2, q2)
    if m == "pchip": return bd_rate(r1, q1, r2, q2) if computable_std(q1, q2) else np.nan
    if m == "akima": return bd_akima(r1, q1, r2, q2)
    return bd_iso(r1, q1, r2, q2)


def to_dom(q, q0, dom):
    """returns transformed quality and number of clamped points"""
    if dom == "map": return q, 0
    u = q / q0; k = int(np.sum((u >= 1 - EPS) | (u <= EPS))); u = np.clip(u, EPS, 1 - EPS)
    return np.log(u / (1 - u)), k


def bpp_rows(det, keys, Bm):
    D = draws("rest", det); N = Bm.shape[0]; out = {}
    for k in keys:
        specs = D[k][:NB] if k.startswith("boot") else D[k]
        out[k] = np.array([(np.bincount(p, minlength=N) @ Bm) / len(p) if kind == "boot" else Bm[p].mean(0) for p, kind in specs])
    return out


def rng(v):
    v = [x for x in v if not np.isnan(x)]; return float(max(v) - min(v)) if len(v) >= 2 else None


def hw(v):
    v = np.asarray(v); v = v[~np.isnan(v)]
    return float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2) if v.size >= 20 else None


def yolo_on_rt(det, ops, specs):
    """yolo mAP x100 on the rtdetr subset (full + the given bootstrap specs over subset positions); cached."""
    from p1_fastmap import build, mapw_fast
    f = E1 / f"yolo_on_rt_{det}.npz"
    if f.exists(): z = np.load(f); return z["full"], z["boot"]
    s = selection(); ids = s["ids"]; pos = {i: k for k, i in enumerate(ids)}; rt = np.array([pos[i] for i in s["rtdetr_subset"]]); N = len(ids)
    full = np.zeros(len(ops)); boot = np.zeros((len(specs), len(ops)))
    for j, o in enumerate(ops):
        z = build(det, o, ids, "full"); w = np.zeros(N); w[rt] = 1; full[j] = 100 * mapw_fast(z, w)
        for b, (p, _) in enumerate(specs): boot[b, j] = 100 * mapw_fast(z, np.bincount(rt[p], minlength=N).astype(float))
        print(" ", det, o, flush=True)
    E1.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, full=full, boot=boot); return full, boot


def main():
    t0 = time.time(); out = {"label": "exploratory (COCO), S1/E_STEP1_H44.md", "eps_logit": EPS, "n_boot": NB, "cells": {}}
    data = {}
    for det in ("yolo26n", "yolo26s", "rtdetr_l"):
        ops, M = load(det, "rest"); bops, Bm = bpp_table(det); assert bops == ops[1:]
        fk = "full|1500" if det == "rtdetr_l" else "full"
        bk = [f"boot|{n}" for n in NS[det]] + (["boot|4952"] if det != "rtdetr_l" else ["boot|1500"])
        BP = bpp_rows(det, [fk] + bk, Bm)
        data[det] = {"ops": ops, "M": {k: M[k][:NB] for k in [fk] + bk}, "BP": BP, "fk": fk, "Bm": Bm}
        print(det, "loaded", round(time.time() - t0), flush=True)

    def curves(det, a, b, key, j, M=None, BP=None):
        d = data[det]; ops = d["ops"]; M = d["M"] if M is None else M; BP = d["BP"] if BP is None else BP
        ca, cb = curve_cols(ops, a), curve_cols(ops, b)
        return BP[key][j, np.array(ca) - 1], M[key][j, ca], BP[key][j, np.array(cb) - 1], M[key][j, cb], M[key][j, 0]

    # B1-B3 full set
    for a, b, det in CELLS:
        c = f"{a}->{b}|{det}"; r1, q1, r2, q2, q0 = curves(det, a, b, data[det]["fk"], 0); E = {"full": {}, "clamped": {}}
        for g, lv in GRIDS.items():
            for dom in DOMAINS:
                x1, k1 = to_dom(q1[lv], q0, dom); x2, k2 = to_dom(q2[lv], q0, dom); E["clamped"][f"{dom}|{g}"] = k1 + k2
                for m in METHODS: E["full"][f"{m}|{dom}|{g}"] = bd(m, r1[lv], x1, r2[lv], x2)
        F = E["full"]
        for g in ("L8", "L4"):
            E[f"range_interp_{g}"] = rng([F[f"{m}|{dom}|{g}"] for m in METHODS for dom in DOMAINS])
            E[f"range_interp_{g}_mapdomain"] = rng([F[f"{m}|map|{g}"] for m in METHODS])
            E[f"n_computable_{g}"] = int(sum(not np.isnan(F[f"{m}|{dom}|{g}"]) for m in METHODS for dom in DOMAINS))
        E["range_grid_pchip"] = rng([F[f"pchip|map|{g}"] for g in GRIDS]); E["range_grid_iso"] = rng([F[f"iso|map|{g}"] for g in GRIDS])
        gr = [x for x in (E["range_grid_pchip"], E["range_grid_iso"]) if x is not None]; E["range_grid"] = max(gr) if gr else None
        allv = np.array([v for v in F.values() if not np.isnan(v)]); E["sd_all_combos"] = float(np.std(allv, ddof=1)); E["n_combos"] = int(allv.size)
        out["cells"][c] = E
    print("full done", round(time.time() - t0), flush=True)

    # B4 sampling
    for a, b, det in CELLS:
        c = f"{a}->{b}|{det}"; S = {}
        for n in NS[det]:
            for g in ("L4", "L8"):
                lv = GRIDS[g]; V = {m: [] for m in METHODS}
                for j in range(NB):
                    r1, q1, r2, q2, _ = curves(det, a, b, f"boot|{n}", j)
                    for m in METHODS: V[m].append(bd(m, r1[lv], q1[lv], r2[lv], q2[lv]))
                for m in METHODS:
                    v = np.array(V[m]); ok = ~np.isnan(v)
                    S[f"{m}|{g}|{n}"] = {"frac_computable": float(ok.mean()), "sd": float(np.std(v[ok], ddof=1)) if ok.sum() >= 20 else None, "hw": hw(v)}
        out["cells"][c]["sampling"] = S
        print(c, "sampling", round(time.time() - t0), flush=True)

    # B5 detectors
    DT = {}
    ny = 4952
    for a, b in CORE_PAIRS:
        for m in ("iso", "poly"):
            for g in ("L8", "L4"):
                lv = GRIDS[g]
                f = lambda det, j, key: (lambda r1, q1, r2, q2, _: bd(m, r1[lv], q1[lv], r2[lv], q2[lv]))(*curves(det, a, b, key, j))
                full = f("yolo26n", 0, "full") - f("yolo26s", 0, "full")
                dv = np.array([f("yolo26n", j, f"boot|{ny}") - f("yolo26s", j, f"boot|{ny}") for j in range(NB)])
                DT[f"{a}->{b}|yolo26n-yolo26s|{m}|{g}"] = {"diff_full": full, "ci95": [float(np.nanpercentile(dv, 2.5)), float(np.nanpercentile(dv, 97.5))]}
    rt = data["rtdetr_l"]; specs = draws("rest", "rtdetr_l")["boot|1500"][:NB]
    for det in ("yolo26n", "yolo26s"):
        yops = data[det]["ops"]; full, boot = yolo_on_rt(det, rt["ops"], specs)
        Mrt = {"full": full[None, :], "boot": boot}; BPrt = {"full": rt["BP"]["full|1500"], "boot": rt["BP"]["boot|1500"]}
        for a, b in CORE_PAIRS:
            for m in ("iso", "poly"):
                for g in ("L8", "L4"):
                    lv = GRIDS[g]
                    f_rt = lambda key, j: (lambda r1, q1, r2, q2, _: bd(m, r1[lv], q1[lv], r2[lv], q2[lv]))(*curves("rtdetr_l", a, b, key, j))
                    f_y = lambda key, j: (lambda r1, q1, r2, q2, _: bd(m, r1[lv], q1[lv], r2[lv], q2[lv]))(*curves(det, a, b, key, j, Mrt, BPrt))
                    fd = f_rt("full|1500", 0) - f_y("full", 0)
                    dv = np.array([f_rt("boot|1500", j) - f_y("boot", j) for j in range(NB)])
                    DT[f"{a}->{b}|rtdetr_l-{det}|{m}|{g}"] = {"diff_full": fd, "yolo_on_1500_full": f_y("full", 0),
                                                              "ci95": [float(np.nanpercentile(dv, 2.5)), float(np.nanpercentile(dv, 97.5))]}
    out["detector"] = DT; print("detector done", round(time.time() - t0), flush=True)

    # B6 budget
    B = {}
    for a, b, det in CELLS:
        c = f"{a}->{b}|{det}"; E = out["cells"][c]
        comps = [f"{a}->{b}|yolo26n-yolo26s|iso|L8"] if det in ("yolo26n", "yolo26s") and b != "hevc" else []
        if b != "hevc": comps += [f"{a}->{b}|rtdetr_l-{y}|iso|L8" for y in ("yolo26n", "yolo26s") if det in (y, "rtdetr_l")]
        detc = max(abs(DT[k]["diff_full"]) for k in comps) if comps else None
        for n in NS[det]:
            row = {"interp_L8": E["range_interp_L8"], "grid": E["range_grid"], "sampling_hw_iso_L8": E["sampling"][f"iso|L8|{n}"]["hw"], "detector": detc}
            v = {k: x for k, x in row.items() if x is not None}; row["largest"] = max(v, key=v.get)
            B[f"{c}|{n}"] = row
    out["budget"] = B
    G = {}
    for a, b in CORE_PAIRS:
        e = [out["cells"][f"{a}->{b}|{d}"] for d in ("yolo26n", "yolo26s")]
        mi = float(np.mean([x["range_interp_L8"] for x in e])); mg = float(np.mean([x["range_grid"] for x in e]))
        ms = float(np.mean([x["sampling"]["iso|L8|500"]["hw"] for x in e]))
        G[f"{a}->{b}"] = {"interp_L8": mi, "grid": mg, "sampling_hw_500": ms, "pass": bool((mi >= 5 or mg >= 5) and ms >= 3)}
    out["gate_GE1"] = {"per_pair": G, "n_pass": int(sum(v["pass"] for v in G.values())), "PASS": bool(sum(v["pass"] for v in G.values()) >= 2)}
    out["sec"] = round(time.time() - t0)
    save_json(out, RES / "e1_budget.json"); print(out["gate_GE1"], out["sec"], flush=True)


if __name__ == "__main__":
    main()
