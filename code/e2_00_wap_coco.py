"""E step 2 part A (exploratory, COCO): wAP vs mAP sampling half-width of BD (isotonic, L8) from stored data.
Draws: first 300 of the stored P1 bootstrap draws boot|n (seed P1+30), n in {166, 500, 1000}; yolo26n/s; 3 core pairs.
Check: fast wAP/mAP vs pycocotools per-class on the full set (|diff| <= 1e-9).
Output results/e2_00_wap_coco.json; cache WORK/p1/e2/wap_<det>.npz.  Usage: python e2_00_wap_coco.py"""
import time, warnings
import numpy as np
from p1_common import *
from p1_fastmap import build, evaluate
from p1_04_draws import draws
from p1_06_stats import bpp_table
from p1_bd import bd_iso
from e2_fastwap import map_wap, pycoco_map_wap

warnings.simplefilter("ignore")
NB = 300; NS = [166, 500, 1000]; DETS = ["yolo26n", "yolo26s"]
PAIRS = [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]
OPS = ["orig"] + OPS_CORE
E2 = P1 / "e2"


def hw(v):
    v = np.asarray(v); v = v[~np.isnan(v)]
    return float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2) if v.size >= 20 else None


def compute(det):
    f = E2 / f"wap_{det}.npz"
    if f.exists(): z = np.load(f); return {k: z[k] for k in z.files}
    ids = selection()["ids"]; N = len(ids); D = draws("rest", det)
    keys = ["full"] + [f"boot|{n}" for n in NS]; out = {}
    for k in keys:
        out[f"map|{k}"] = np.zeros((1 if k == "full" else NB, len(OPS))); out[f"wap|{k}"] = np.zeros_like(out[f"map|{k}"])
    for j, o in enumerate(OPS):
        z = build(det, o, ids, "full")
        for k in keys:
            for b, (p, kind) in enumerate(D[k][:1 if k == "full" else NB]):
                w = np.bincount(p, minlength=N).astype(float) if kind == "boot" else (lambda w: (w.__setitem__(p, 1.0), w)[1])(np.zeros(N))
                m, wa = map_wap(z, w); out[f"map|{k}"][b, j] = 100 * m; out[f"wap|{k}"][b, j] = 100 * wa
        print(det, o, flush=True)
    E2.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **out); return out


def main():
    t0 = time.time(); R = {"label": "exploratory (COCO), S1/E_STEP2_H44.md part A", "n_boot": NB, "check": [], "cells": {}}
    ids = selection()["ids"]
    for det, op in [("yolo26n", "orig"), ("yolo26n", "jpeg_3"), ("yolo26s", "avif_5")]:
        a_m, a_w, _, _ = pycoco_map_wap(evaluate(det, op, ids)); f_m, f_w = map_wap(build(det, op, ids, "full"), np.ones(len(ids)))
        R["check"].append({"det": det, "op": op, "pycoco_map": a_m, "fast_map": f_m, "pycoco_wap": a_w, "fast_wap": f_w,
                           "absdiff_map": abs(a_m - f_m), "absdiff_wap": abs(a_w - f_w)}); print(R["check"][-1], flush=True)
    R["check_pass"] = bool(max(max(c["absdiff_map"], c["absdiff_wap"]) for c in R["check"]) <= 1e-9)
    for det in DETS:
        V = compute(det); bops, Bm = bpp_table(det); Bm = Bm[:, [bops.index(o) for o in OPS[1:]]]
        D = draws("rest", det); N = Bm.shape[0]
        st = np.load(P1 / "maps" / "rest" / det / "orig.npz")["boot|500"][:NB]
        R[f"maxdiff_vs_stored_map_orig_boot500|{det}"] = float(np.max(np.abs(st - V["map|boot|500"][:, 0])))
        BP = {k: np.array([(np.bincount(p, minlength=N) @ Bm) / len(p) if kind == "boot" else Bm[p].mean(0) for p, kind in D[k][:1 if k == "full" else NB]])
              for k in ["full"] + [f"boot|{n}" for n in NS]}
        for a, b in PAIRS:
            ca = [OPS.index(f"{a}_{l}") for l in range(8)]; cb = [OPS.index(f"{b}_{l}") for l in range(8)]
            c = f"{a}->{b}|{det}"; E = {}
            for met in ("map", "wap"):
                f = lambda k, j: bd_iso(BP[k][j, np.array(ca) - 1], V[f"{met}|{k}"][j, ca], BP[k][j, np.array(cb) - 1], V[f"{met}|{k}"][j, cb])
                E[f"bd_full_{met}"] = f("full", 0)
                for n in NS:
                    v = np.array([f(f"boot|{n}", j) for j in range(NB)]); E[f"hw_{met}|{n}"] = hw(v); E[f"frac_nan_{met}|{n}"] = float(np.isnan(v).mean())
            for n in NS: E[f"r|{n}"] = E[f"hw_wap|{n}"] / E[f"hw_map|{n}"]
            R["cells"][c] = E; print(c, {k: round(v, 3) for k, v in E.items() if v is not None}, flush=True)
    R["r_COCO_500"] = float(np.median([E["r|500"] for E in R["cells"].values()]))
    for n in NS: R[f"r_median|{n}"] = float(np.median([E[f"r|{n}"] for E in R["cells"].values()]))
    R["sec"] = round(time.time() - t0); save_json(R, RES / "e2_00_wap_coco.json"); print("r_COCO(500)", R["r_COCO_500"], R["sec"], flush=True)


if __name__ == "__main__":
    main()
