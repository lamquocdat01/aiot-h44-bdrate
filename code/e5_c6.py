"""E-FINISH H0 — claim C6 "BD-rate vs. operating-point (quality-level) decision".
Preregistered in prereg/PREREG_H44_E_C6.md (frozen in prereg/FREEZE_E_C6.txt before any q* was computed on VOC).
Dataset 'voc' = CONFIRMATORY (gates C6a, C6b). Dataset 'coco' = EXPLORATORY (same quantities as results/z1_c.json, recomputed
by this code for the side-by-side table). Optional extra detectors (e.g. frcnn_mv3, H1) = EXPLORATORY, reported only, never gated.

Definitions (copied from the prereg):
  subset   = rtdetr_l 1,500-image subset of the dataset; every detector evaluated on it (rtfull)
  D_d(c,q) = mAP_d(orig) - mAP_d(c,q), mAP = AP@[.5:.95] x 100 (fast COCOeval npz)
  q*_d(c)  = lowest q in 0..7 with D_d(c,q) <= delta at q and at every higher q; none -> undefined
  CI       = 95 % percentile CI of D from the stored paired bootstrap 'rtboot|1500' (1,000 draws, same image indices for every
             detector, the original and every level); no new draws
  C6a      = within YOLO (yolo26n/s/l, 3 pairs x 3 codecs = 9 cells): |q*_A - q*_B| <= 1; PASS if >= 8/9 cells (undefined = fail)
  C6b      = cross-family yolo26n <-> rtdetr_l, 3 codecs x 2 directions = 6 cells: overshoot D_B(q*_A) - delta <= 1.0 in all 6 cells
             (point estimate) AND number of sure violations (CI of D_B(q*_A) entirely above delta) <= 1; undefined = fail
  delta    = 1 (gated); delta = 2 reported. bpp(q*_A)/bpp(q*_B) and BD-rate (iso L8) differences reported.
Usage: python e5_c6.py voc | coco | extra <det> [<det>...]   (extra appends to results/e5_c6.json under 'exploratory_extra')"""
import sys
import numpy as np
from e3_common import *
from e3_data import Data, BOPS

YOLO = ["yolo26n", "yolo26s", "yolo26l"]; DETS = YOLO + ["rtdetr_l"]
E5 = P1 / "e5"; OUT = RES / "e5_c6.json"


def orig_fast(ds, det, D):
    """(npz, sample positions) for the original image of det on the subset"""
    if det == "rtdetr_l": return fast(ds, det, "orig", "rt"), np.arange(len(D.rt))
    if det in ("yolo26n", "yolo26s"): return fast(ds, det, "orig", "full"), D.rt
    z = np.load(P1 / "e3" / "fast" / ds / det / "orig.npz"); return {k: z[k] for k in z.files}, np.arange(len(D.rt))


def orig_boot(ds, det, D):
    f = E5 / "maps" / ds / f"{det}_orig.npz"
    if f.exists(): z = np.load(f); return float(z["full"][0]), z["boot"]
    inv = {p: k for k, p in enumerate(D.rt)}; M = len(D.rt)
    W = np.vstack([np.ones(M)] + [np.bincount([inv[p] for p in s], minlength=M) for s in D.specs("rtboot|1500")]).astype(float)
    z, smp = orig_fast(ds, det, D)
    v = 100 * batch_map_wap(z, smp, W, THREADS)[:, 0]; f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, full=v[:1], boot=v[1:])
    return float(v[0]), v[1:]


def qstar(dfull, dl):
    ok = [x <= dl for x in dfull]
    for q in range(8):
        if all(ok[q:]): return q
    return None


def tables(ds, dets):
    D = Data(ds); bp = dict(zip(BOPS, D.BP("rtfull")[0])); Dl = {}
    for det in dets:
        of, ob = orig_boot(ds, det, D)
        assert abs(of - D.Q(det, "orig", "map", "rtfull")[0]) < 1e-9, (ds, det)
        for c in CORE:
            for q in range(8):
                op = f"{c}_{q}"; Dl[(det, c, q)] = (of - D.Q(det, op, "map", "rtfull")[0], ob - D.Q(det, op, "map", "rtboot|1500"))
    bd = {det: {f"{a}->{b}": float(D.bd("iso", det, a, b, "map", "rtfull", 0)) for a, b in PAIRS} for det in dets}
    return D, bp, Dl, bd


def cell(Dl, bp, A, B, c, qa, qb, dl):
    if qa is None or qb is None: return {"qA": qa, "qB": qb, "defined": False}
    dA, bA = Dl[(B, c, qa)]; dB, bB = Dl[(B, c, qb)]
    ciA = ci(bA); ciB = ci(bB)
    return {"qA": qa, "qB": qb, "defined": True, "dB_at_qA": float(dA), "ci_dB_at_qA": ciA, "overshoot": float(dA - dl),
            "sure_violation": bool(ciA[0] > dl), "dB_at_qB": float(dB), "ci_dB_at_qB": ciB,
            "bpp_qA": float(bp[f"{c}_{qa}"]), "bpp_qB": float(bp[f"{c}_{qb}"]), "bpp_ratio_qA_over_qB": float(bp[f"{c}_{qa}"] / bp[f"{c}_{qb}"])}


def analyse(ds, dets, gated):
    D, bp, Dl, bd = tables(ds, dets)
    R = {"dataset": ds, "evidence": "confirmatory (prereg/PREREG_H44_E_C6.md)" if gated else "exploratory",
         "subset": "rtdetr_l 1,500-image subset; paired rtboot|1500 draws", "detectors": dets,
         "orig_map": {d: float(D.Q(d, "orig", "map", "rtfull")[0]) for d in dets},
         "delta_full": {f"{d}|{c}": [float(Dl[(d, c, q)][0]) for q in range(8)] for d in dets for c in CORE},
         "bpp_rtfull": {f"{c}_{q}": float(bp[f"{c}_{q}"]) for c in CORE for q in range(8)},
         "bd_iso_L8_rtfull": bd, "qstar": {}, "C6a": {}, "C6b": {}, "all_cells": {}}
    for dl in (1, 2):
        qs = {(d, c): qstar([Dl[(d, c, q)][0] for q in range(8)], dl) for d in dets for c in CORE}
        R["qstar"][str(dl)] = {f"{d}|{c}": qs[(d, c)] for d in dets for c in CORE}
        for A in dets:
            for B in dets:
                if A != B:
                    for c in CORE: R["all_cells"][f"{A}->{B}|{c}|{dl}"] = cell(Dl, bp, A, B, c, qs[(A, c)], qs[(B, c)], dl)
        # C6a within YOLO (unordered pairs)
        ya = [d for d in YOLO if d in dets]; rows = []
        for i in range(len(ya)):
            for j in range(i + 1, len(ya)):
                for c in CORE:
                    qa, qb = qs[(ya[i], c)], qs[(ya[j], c)]; ok = qa is not None and qb is not None and abs(qa - qb) <= 1
                    rows.append({"pair": f"{ya[i]}|{ya[j]}", "codec": c, "qA": qa, "qB": qb, "ok": bool(ok),
                                 "bd_diff_B_minus_A": bd[ya[j]][f"jpeg->{c}"] - bd[ya[i]][f"jpeg->{c}"] if c != "jpeg" else None})
        k = sum(r["ok"] for r in rows)
        R["C6a"][str(dl)] = {"cells": rows, "n_ok": k, "n": len(rows), "pass": bool(k >= 8) if len(rows) == 9 else None}
        # C6b cross family
        rows = []
        for A, B in (("yolo26n", "rtdetr_l"), ("rtdetr_l", "yolo26n")):
            for c in CORE:
                x = R["all_cells"][f"{A}->{B}|{c}|{dl}"]; rows.append({"A": A, "B": B, "codec": c, **x})
        ov_ok = all(r["defined"] and r["overshoot"] <= 1.0 for r in rows); ns = sum(r.get("sure_violation", False) for r in rows)
        R["C6b"][str(dl)] = {"cells": rows, "max_overshoot": max((r["overshoot"] for r in rows if r["defined"]), default=None),
                             "all_overshoot_le_1": bool(ov_ok), "n_sure_violation": int(ns), "pass": bool(ov_ok and ns <= 1)}
    if gated:
        R["verdict"] = {"C6a": "DAT" if R["C6a"]["1"]["pass"] else "KHONG DAT", "C6b": "DAT" if R["C6b"]["1"]["pass"] else "KHONG DAT"}
    R["bd_gap_rt_minus_n"] = {p: bd["rtdetr_l"][p] - bd["yolo26n"][p] for p in bd["yolo26n"]}
    return R


def extra(newdets):
    """exploratory: q* (delta 1, 2) of extra detectors on both datasets, next to the gated detectors"""
    out = {}
    for ds in ("voc", "coco"):
        if not all((P1 / "e3" / "maps" / ds / d / "avif_7.npz").exists() for d in newdets): continue
        R = analyse(ds, DETS + list(newdets), gated=False)
        out[ds] = {"orig_map": R["orig_map"], "qstar": R["qstar"], "bd_iso_L8_rtfull": R["bd_iso_L8_rtfull"],
                   "transfer": {k: v for k, v in R["all_cells"].items() if any(k.startswith(f"{d}->") or f"->{d}|" in k for d in newdets)},
                   "delta_full": {k: v for k, v in R["delta_full"].items() if k.split("|")[0] in newdets}}
    return out


def main():
    a = sys.argv[1:]; prev = load_json(OUT) if OUT.exists() else {}
    if a[0] in ("voc", "coco"):
        prev[a[0]] = analyse(a[0], DETS, gated=(a[0] == "voc"))
        r = prev[a[0]]; print(a[0], "qstar d1", r["qstar"]["1"]); print("C6a", r["C6a"]["1"]["n_ok"], "/", r["C6a"]["1"]["n"], r["C6a"]["1"]["pass"])
        print("C6b max overshoot", r["C6b"]["1"]["max_overshoot"], "sure", r["C6b"]["1"]["n_sure_violation"], r["C6b"]["1"]["pass"], r.get("verdict"))
    elif a[0] == "extra":
        prev["exploratory_extra"] = {"label": "EXPLORATORY (H1, not preregistered for C6)", "detectors": a[1:], **extra(a[1:])}
        for ds, r in prev["exploratory_extra"].items():
            if isinstance(r, dict) and "qstar" in r: print(ds, r["qstar"]["1"])
    save_json(prev, OUT)


if __name__ == "__main__":
    main()
