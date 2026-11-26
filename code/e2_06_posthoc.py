"""E step 2 POST HOC (written after seeing VOC numbers; prereg/AMENDMENT_LOG_E.md). Not part of the frozen analysis.
(1) COCO (exploratory) vs VOC (confirmatory) for each K quantity; (2) full-set JPEG/WebP/AVIF mAP curves on VOC (why PCHIP is not
computable for JPEG pairs); (3) K1 part 2 with isotonic-adjusted curves for PCHIP/Akima (range over methods after isotonic step).
Output results/e2_posthoc.json.  Usage: python e2_06_posthoc.py"""
import numpy as np
from e2_voc_common import *
from p1_bd import iso_curve, bd_rate
from e2_05_confirm import bd_akima

C = load_json(RES / "e2_confirm.json"); E1 = load_json(RES / "e1_budget.json"); A = load_json(RES / "e2_00_wap_coco.json")
PAIRS = ["jpeg->webp", "jpeg->avif", "webp->avif"]; DETS = ["yolo26n", "yolo26s"]
out = {"label": "POST HOC (after VOC numbers)", "coco_vs_voc": {}}
for d in DETS:
    for p in PAIRS:
        c = f"{p}|{d}"; e = E1["cells"][c]; v = C["cells"][c]
        out["coco_vs_voc"][c] = {n: {"hw_map_coco": e["sampling"][f"iso|L8|{n}"]["hw"], "hw_map_voc": v[f"hw|map|{n}"],
                                     "r_coco": A["cells"][c].get(f"r|{n}"), "r_voc": v[f"hw|wap|{n}"] / v[f"hw|map|{n}"]} for n in (166, 500, 1000)}
        out["coco_vs_voc"][c]["range_fit_mapdomain_L8_coco_pchip_akima_iso"] = (lambda F: (lambda x: max(x) - min(x) if len(x) >= 2 else None)(
            [F[f"{m}|map|L8"] for m in ("pchip", "akima", "iso") if F[f"{m}|map|L8"] == F[f"{m}|map|L8"]]))(e["full"])
out["detector_coco"] = {k: E1["detector"][k]["diff_full"] for k in E1["detector"] if k.endswith("|iso|L8")}
# (2) curves and (3) isotonic-adjusted fits
z = np.load(WV / "bpp.npz"); ids = ids_all(); pos = {i: k for k, i in enumerate(ids)}; Bsum = {}
for o, b in zip(z["op"], z["bpp"]): Bsum[str(o)] = Bsum.get(str(o), 0.0) + b
cur = {}
for d in DETS:
    for c in CORE:
        q = [float(np.load(WV / "maps" / d / f"{c}_{l}.npz")["map|full"][0]) for l in range(8)]
        r = [Bsum[f"{c}_{l}"] / len(ids) for l in range(8)]; cur[f"{d}|{c}"] = {"bpp": r, "map": q, "diffs": np.diff(q).tolist()}
out["full_curves_voc"] = cur
K1b = {}
for d in DETS:
    for p in PAIRS:
        a, b = p.split("->"); A1 = cur[f"{d}|{a}"]; B1 = cur[f"{d}|{b}"]
        r1, q1 = iso_curve(np.array(A1["bpp"]), np.array(A1["map"])); r2, q2 = iso_curve(np.array(B1["bpp"]), np.array(B1["map"]))
        v = {"iso(pchip on iso)": bd_rate(r1, q1, r2, q2), "akima on iso": bd_akima(r1, q1, r2, q2) if min(len(q1), len(q2)) >= 2 else np.nan}
        ok = [x for x in v.values() if x == x]
        K1b[f"{p}|{d}"] = {**v, "range": (max(ok) - min(ok)) if len(ok) >= 2 else None, "n_points_after_iso": [int(len(q1)), int(len(q2))],
                           "hw500_voc": C["cells"][f"{p}|{d}"]["hw|map|500"]}
out["K1_part2_isotonic_adjusted"] = K1b
save_json(out, RES / "e2_posthoc.json")
for k, v in cur.items(): print(k, [round(x, 2) for x in v["map"]])
for k, v in K1b.items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()})
for k, v in out["coco_vs_voc"].items(): print(k, v)
print(out["detector_coco"])
