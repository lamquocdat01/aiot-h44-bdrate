"""S2FIX F2 + F3 (post hoc; definitions S2/E_S2FIX_H44.md 0). Output results/e3_f2.json, results/e3_f3.json.
Usage: python e3_03_f2f3.py [f2|f3|both]"""
import sys, time
import numpy as np
from scipy import stats
from e3_data import *

DETS = ["yolo26n", "yolo26s"]; NS_B = [50, 100, 166, 500, 1000, 2500]


def k_quantities(D, lv):
    out = {"K1": {}, "K2": {}, "K3a": {}, "K3b": {}, "K4": {}}
    hwc = {}
    for det in DETS:
        for a, b in PAIRS:
            c = f"{a}->{b}|{det}"
            for n in (166, 500):
                hwc[(c, "map", n)] = hw(D.bdv("iso", det, a, b, "map", f"boot|{n}", lv))
            hwc[(c, "wap", 500)] = hw(D.bdv("iso", det, a, b, "wap", "boot|500", lv))
            fv = [D.bd(m, det, a, b, "map", "full", 0, lv) for m in ("pchip", "akima", "iso")]; ok = [x for x in fv if x == x]
            rg = float(max(ok) - min(ok)) if len(ok) >= 2 else None; h = hwc[(c, "map", 500)]
            out["K1"][c] = {"hw500": h, "range": rg, "tested": rg is not None, "pass": bool(h is not None and h >= 3 and (rg is None or h > rg))}
            out["K4"][c] = hwc[(c, "wap", 500)] / h
    for det in DETS:
        for n in (166, 500):
            r = hwc[(f"webp->avif|{det}", "map", n)] / hwc[(f"jpeg->webp|{det}", "map", n)]; out["K2"][f"{det}|{n}"] = {"ratio": r, "pass": bool(r >= 1.5)}
    for a, b in PAIRS:
        d0 = D.bd("iso", "rtdetr_l", a, b, "map", "rtfull", 0, lv) - D.bd("iso", "yolo26n", a, b, "map", "rtfull", 0, lv)
        dv = D.bdv("iso", "rtdetr_l", a, b, "map", "rtboot|1500", lv) - D.bdv("iso", "yolo26n", a, b, "map", "rtboot|1500", lv); cc = ci(dv)
        out["K3a"][f"{a}->{b}"] = {"diff": d0, "ci95": cc, "pass": bool(abs(d0) >= 8 and cc is not None and (cc[0] > 0 or cc[1] < 0)), "registered": a == "jpeg"}
        x = D.bd("iso", "yolo26n", a, b, "map", "full", 0, lv); y = D.bd("iso", "yolo26s", a, b, "map", "full", 0, lv)
        out["K3b"][f"{a}->{b}"] = {"absdiff": abs(x - y), "pass": bool(abs(x - y) <= 6)}
    out["summary"] = {"K1_pass": sum(v["pass"] for v in out["K1"].values()), "K1_tested": sum(v["tested"] for v in out["K1"].values()),
                      "K1": sum(v["pass"] for v in out["K1"].values()) >= 5, "K2_pass": sum(v["pass"] for v in out["K2"].values()),
                      "K2": sum(v["pass"] for v in out["K2"].values()) >= 3,
                      "K3": all(v["pass"] for v in out["K3a"].values() if v["registered"]) and all(v["pass"] for v in out["K3b"].values()),
                      "K4_median_r": float(np.median(list(out["K4"].values()))), "K4": bool(np.median(list(out["K4"].values())) <= 0.90)}
    return out


def need_n(ns, h, target):
    for i in range(len(ns) - 1):
        if h[i] > target >= h[i + 1]:
            t = (np.log(target) - np.log(h[i])) / (np.log(h[i + 1]) - np.log(h[i])); return float(np.exp(np.log(ns[i]) + t * (np.log(ns[i + 1]) - np.log(ns[i]))))
    if h[0] <= target: return "<= 50"
    return "> 2500"


def f2():
    t0 = time.time(); R = {"label": "POST HOC F2", "grids": {}, "n_curve": {}}
    for ds in ("coco", "voc"):
        D = Data(ds)
        for g, lv in GRIDS.items():
            R["grids"][f"{ds}|{g}"] = k_quantities(D, lv); print(ds, g, R["grids"][f"{ds}|{g}"]["summary"], round(time.time() - t0), flush=True)
        for det in DETS:
            for a, b in PAIRS:
                c = f"{a}->{b}|{det}"; h, cis = [], {}
                for n in NS_B:
                    v = D.bdv("iso", det, a, b, "map", f"boot|{n}"); h.append(hw(v)); cis[n] = ci(v)
                x, y = np.log(NS_B), np.log(h); res = stats.linregress(x, y); tq = stats.t.ppf(0.975, len(NS_B) - 2)
                excl = [n for n in NS_B if cis[n][0] > 0 or cis[n][1] < 0]
                R["n_curve"][f"{ds}|{c}"] = {"hw": dict(zip(NS_B, h)), "ci": cis, "a": -res.slope, "a_ci95": [-res.slope - tq * res.stderr, -res.slope + tq * res.stderr],
                                             "c": float(np.exp(res.intercept)), "r2": res.rvalue ** 2, "n_for_hw5": need_n(NS_B, h, 5.0), "n_for_hw2": need_n(NS_B, h, 2.0),
                                             "smallest_n_ci_excl0": min(excl) if excl else None}
                print(ds, c, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in R["n_curve"][f"{ds}|{c}"].items() if k in ("a", "n_for_hw5", "n_for_hw2", "smallest_n_ci_excl0")}, flush=True)
    for ds in ("coco", "voc"):
        base = R["grids"][f"{ds}|L8"]["summary"]
        R[f"changed_vs_L8|{ds}"] = {g: [k for k in ("K1", "K2", "K3", "K4") if R["grids"][f"{ds}|{g}"]["summary"][k] != base[k]] for g in GRIDS}
    R["sec"] = round(time.time() - t0); save_json(R, RES / "e3_f2.json")


def f3():
    t0 = time.time(); R = {"label": "POST HOC F3", "cells": {}, "class_share_top5": {}, "ranking": {}}
    cells = [(a, b, d) for d in DETS for a, b in PAIRS] + [(c, "hevc", "yolo26n") for c in CORE]
    for ds in ("coco", "voc"):
        D = Data(ds)
        for a, b, det in cells:
            fm = D.bd("iso", det, a, b, "map", "full", 0); fw = D.bd("iso", det, a, b, "wap", "full", 0)
            dv = D.bdv("iso", det, a, b, "wap", "boot|4952") - D.bdv("iso", det, a, b, "map", "boot|4952")
            R["cells"][f"{ds}|{a}->{b}|{det}"] = {"bd_map": fm, "bd_wap": fw, "diff": fw - fm, "diff_ci95": ci(dv), "sign_flip": bool(np.sign(fm) != np.sign(fw))}
        for det in DETS:
            cs = ["webp", "avif"] + (["hevc"] if det == "yolo26n" else [])
            rk = {met: sorted(cs, key=lambda c: R["cells"][f"{ds}|jpeg->{c}|{det}"][f"bd_{met}"]) for met in ("map", "wap")}
            R["ranking"][f"{ds}|{det}"] = {**rk, "changed": rk["map"] != rk["wap"]}
        z = fast(ds, "yolo26n", "orig"); nk = z["npig"].sum(0).astype(float)
        if ds == "coco":
            from p1_fastmap import gt
            G = gt(); cid = sorted(G.getCatIds()); names = [G.cats[c]["name"] for c in cid]
        else:
            g = V.build_gt(); names = [c["name"] for c in sorted(g["categories"], key=lambda c: c["id"])]
        o = np.argsort(-nk)[:5]; R["class_share_top5"][ds] = {"classes": [names[k] for k in o], "share": (nk[o] / nk.sum()).tolist(), "n_classes": int((nk > 0).sum()),
                                                            "total_gt": int(nk.sum()), "share_top1": float(nk[o[0]] / nk.sum()), "share_top5": float(nk[o].sum() / nk.sum())}
        print(ds, R["class_share_top5"][ds], round(time.time() - t0), flush=True)
    for k, v in R["cells"].items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
    print(R["ranking"]); R["sec"] = round(time.time() - t0); save_json(R, RES / "e3_f3.json")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "both"
    if w in ("f2", "both"): f2()
    if w in ("f3", "both"): f3()
