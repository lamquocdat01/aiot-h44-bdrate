"""E step 2 D: confirmatory analysis on VOC 2007 test (prereg/PREREG_H44_E.md, hashed in prereg/FREEZE_E.txt).
Inputs: WV/maps/<det>/<op>.npz (e2_04), WV/bpp.npz (e2_01). BD: ln(bpp) vs mAP (or wAP) x100, map domain, grid L8.
"iso" = isotonic regression of each curve then PCHIP BD (p1_bd.bd_iso); "pchip" = standard PCHIP BD, only when both curves are
strictly increasing (p1_bd.computable_std); "akima" = Akima on the same rule. Half-width = (P97.5 - P2.5)/2 of the bootstrap BD.
Output results/e2_confirm.json.  Usage: python e2_05_confirm.py"""
import time, warnings
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from e2_voc_common import *
from p1_bd import bd_rate, bd_iso, computable_std, nonmono

warnings.simplefilter("ignore")
PAIRS = [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]
HEVC_PAIRS = [("jpeg", "hevc"), ("webp", "hevc"), ("avif", "hevc")]
DETS = ["yolo26n", "yolo26s"]
# K4 registered form (set from part A before freezing): "le090" | "band" | "none"
K4_MODE = "le090"


def bd_akima(r1, q1, r2, q2):
    if not computable_std(q1, q2): return np.nan
    lo, hi = max(q1.min(), q2.min()), min(q1.max(), q2.max())
    f1 = Akima1DInterpolator(q1, np.log(r1)); f2 = Akima1DInterpolator(q2, np.log(r2))
    return float((np.exp((f2.integrate(lo, hi) - f1.integrate(lo, hi)) / (hi - lo)) - 1) * 100)


def bd(m, r1, q1, r2, q2):
    if m == "pchip": return bd_rate(r1, q1, r2, q2) if computable_std(q1, q2) else np.nan
    if m == "akima": return bd_akima(r1, q1, r2, q2)
    return bd_iso(r1, q1, r2, q2)


def hw(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2) if v.size >= 20 else None


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size >= 20 else None


def main():
    from e2_04_voc_draws import specs
    t0 = time.time(); ids = ids_all(); N = len(ids); S, rt = specs(N); pos = {i: k for k, i in enumerate(ids)}
    zb = np.load(WV / "bpp.npz"); bops = OPS_CORE + OPS_HEVC; oi = {o: j for j, o in enumerate(bops)}; Bm = np.zeros((N, len(bops)))
    for i, o, b in zip(zb["img"], zb["op"], zb["bpp"]): Bm[pos[int(i)], oi[str(o)]] = b
    assert (Bm > 0).all()
    BP = {k: np.array([np.bincount(p, minlength=N) @ Bm / len(p) for p in v]) for k, v in S.items()}
    Q = {}
    for det in DETS + ["rtdetr_l"]:
        for o in ops_of(det):
            z = np.load(WV / "maps" / det / f"{o}.npz"); Q[(det, o)] = {k: z[k] for k in z.files}

    def curve(det, c, met, key, j):
        cols = [oi[f"{c}_{l}"] for l in range(8)]
        return BP[key][j, cols], np.array([Q[(det, f"{c}_{l}")][f"{met}|{key}"][j] for l in range(8)])

    def bdv(m, det, a, b, met, key, j):
        r1, q1 = curve(det, a, met, key, j); r2, q2 = curve(det, b, met, key, j); return bd(m, r1, q1, r2, q2)

    R = {"label": "confirmatory (PASCAL VOC 2007 test)", "N": N, "n_boot": NB, "K4_MODE": K4_MODE, "cells": {}}
    R["map_orig_full"] = {det: float(Q[(det, "orig")]["map|" + ("rtfull" if det == "rtdetr_l" else "full")][0]) for det in DETS + ["rtdetr_l"]}
    R["bpp_mean_full"] = dict(zip(bops, BP["full"][0].tolist()))
    nkeys = NS + [N]
    for det in DETS:
        for a, b in PAIRS + (HEVC_PAIRS if det == "yolo26n" else []):
            c = f"{a}->{b}|{det}"; E = {}
            for met in ("map", "wap"):
                E[f"full|{met}"] = {m: bdv(m, det, a, b, met, "full", 0) for m in ("pchip", "akima", "iso")}
                for n in nkeys:
                    key = f"boot|{n}"; v = np.array([bdv("iso", det, a, b, met, key, j) for j in range(NB)])
                    E[f"hw|{met}|{n}"] = hw(v); E[f"ci|{met}|{n}"] = ci(v); E[f"fracnan_iso|{met}|{n}"] = float(np.isnan(v).mean())
                    if met == "map":
                        E[f"frac_pchip_computable|{n}"] = float(np.mean([computable_std(curve(det, a, met, key, j)[1], curve(det, b, met, key, j)[1]) for j in range(NB)]))
            ex = [n for n in nkeys if E[f"ci|map|{n}"] and (E[f"ci|map|{n}"][0] > 0 or E[f"ci|map|{n}"][1] < 0)]
            E["smallest_n_ci_excludes_0_map"] = min(ex) if ex else None
            R["cells"][c] = E; print(c, round(time.time() - t0), flush=True)
    R["nonmono"] = {f"{det}|{c}|{n}": float(np.mean([nonmono(curve(det, c, "map", f"boot|{n}", j)[1]) for j in range(NB)]))
                    for det in DETS for c in CORE + (["hevc"] if det == "yolo26n" else []) for n in nkeys}

    # K1
    K1 = {}
    for det in DETS:
        for a, b in PAIRS:
            E = R["cells"][f"{a}->{b}|{det}"]; v = [x for x in E["full|map"].values() if not np.isnan(x)]
            h = E["hw|map|500"]; rg = float(max(v) - min(v)) if len(v) >= 2 else None
            K1[f"{a}->{b}|{det}"] = {"hw500": h, "range_fit_full": rg, "n_methods": len(v), "note": None if rg is not None else "<2 methods: part 2 counted as met",
                                     "pass": bool(h is not None and h >= 3.0 and (rg is None or h > rg))}
    R["K1"] = {"cells": K1, "n_pass": int(sum(x["pass"] for x in K1.values())), "PASS": bool(sum(x["pass"] for x in K1.values()) >= 5)}
    # K2
    K2 = {}
    for det in DETS:
        for n in (166, 500):
            h1 = R["cells"][f"jpeg->webp|{det}"][f"hw|map|{n}"]; h2 = R["cells"][f"webp->avif|{det}"][f"hw|map|{n}"]
            K2[f"{det}|{n}"] = {"hw_jpeg_webp": h1, "hw_webp_avif": h2, "ratio": h2 / h1, "pass": bool(h2 >= 1.5 * h1)}
    R["K2"] = {"cells": K2, "n_pass": int(sum(x["pass"] for x in K2.values())), "PASS": bool(sum(x["pass"] for x in K2.values()) >= 3)}
    # K3
    K3a = {}
    kb = f"rtboot|{N_RT}"
    for a, b in [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]:
        d_full = bdv("iso", "rtdetr_l", a, b, "map", "rtfull", 0) - bdv("iso", "yolo26n", a, b, "map", "rtfull", 0)
        dv = np.array([bdv("iso", "rtdetr_l", a, b, "map", kb, j) - bdv("iso", "yolo26n", a, b, "map", kb, j) for j in range(NB)])
        cc = ci(dv); K3a[f"{a}->{b}"] = {"bd_rtdetr_l": bdv("iso", "rtdetr_l", a, b, "map", "rtfull", 0), "bd_yolo26n_1500": bdv("iso", "yolo26n", a, b, "map", "rtfull", 0),
                                        "diff": d_full, "ci95": cc, "pass": bool(abs(d_full) >= 8 and (cc[0] > 0 or cc[1] < 0)),
                                        "registered": a == "jpeg"}
    K3b = {}
    for a, b in PAIRS:
        x = bdv("iso", "yolo26n", a, b, "map", "full", 0); y = bdv("iso", "yolo26s", a, b, "map", "full", 0)
        dv = np.array([bdv("iso", "yolo26n", a, b, "map", f"boot|{N}", j) - bdv("iso", "yolo26s", a, b, "map", f"boot|{N}", j) for j in range(NB)])
        K3b[f"{a}->{b}"] = {"bd_yolo26n": x, "bd_yolo26s": y, "absdiff": abs(x - y), "ci95_reported": ci(dv), "pass": bool(abs(x - y) <= 6)}
    pa = all(v["pass"] for k, v in K3a.items() if v["registered"]); pb = all(v["pass"] for v in K3b.values())
    R["K3"] = {"family": K3a, "within_yolo": K3b, "part1_pass": bool(pa), "part2_pass": bool(pb), "PASS": bool(pa and pb)}
    # K4
    r = {f"{a}->{b}|{det}": R["cells"][f"{a}->{b}|{det}"]["hw|wap|500"] / R["cells"][f"{a}->{b}|{det}"]["hw|map|500"] for det in DETS for a, b in PAIRS}
    med = float(np.median(list(r.values())))
    if K4_MODE == "le090": k4 = bool(med <= 0.90)
    elif K4_MODE == "band": k4 = bool(0.85 <= med <= 1.15)
    else: k4 = None
    R["K4"] = {"r_cells_500": r, "median_r_500": med, "mode": K4_MODE, "PASS": k4,
               "median_r_by_n": {n: float(np.median([R["cells"][f"{a}->{b}|{det}"][f"hw|wap|{n}"] / R["cells"][f"{a}->{b}|{det}"][f"hw|map|{n}"]
                                                     for det in DETS for a, b in PAIRS])) for n in nkeys}}
    R["sec"] = round(time.time() - t0); save_json(R, RES / "e2_confirm.json")
    print({k: R[k]["PASS"] for k in ("K1", "K2", "K3", "K4")}, R["sec"], flush=True)


if __name__ == "__main__":
    main()
