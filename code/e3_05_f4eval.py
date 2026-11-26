"""S2FIX F4 (exploratory): evaluate extra detectors on the rtdetr_l 1,500-image subset, then apply the F4 rule (S2/E_S2FIX_H44.md 0).
build: COCOeval.evaluate once per (ds, det, op) -> WORK/p1/e3/fast/<ds>/<det>/<op>.npz (same code path as e2_03.build_generic);
       mAP/wAP x100 for rtfull and the SAME rtboot|1500 draws used for rtdetr_l (paired) -> WORK/p1/e3/maps/<ds>/<det>/<op>.npz.
check: fast vs pycocotools on 2 points per (ds, det) (<= 1e-9).
analyse: results/e3_f4.json.  Usage: python e3_05_f4eval.py build <det> <ds...> | analyse <det...>"""
import sys, time
import numpy as np
from e3_common import *
from e3_data import Data, E3M
from e2_03_voc_eval import build_generic, evaluate_generic
from e2_fastwap import map_wap, pycoco_map_wap

E3D = P1 / "e3" / "det"; E3F = P1 / "e3" / "fast"
JPEG_PAIRS = [("jpeg", "webp"), ("jpeg", "avif")]


def G_of(ds):
    if ds == "coco":
        from p1_fastmap import gt
        return gt()
    return V.gt_coco()


def dets(ds, det, op, I):
    keep = None if ds == "coco" else set(c["id"] for c in V.build_gt()["categories"]); have = set(); out = []
    for f in sorted((E3D / ds / det / op).glob("[0-9][0-9][0-9].npz")):
        z = np.load(f); have |= set(int(i) for i in z["imgs"])
        m = np.ones(z["img"].size, bool) if keep is None else np.isin(z["cat"], list(keep))
        for i, c, b, s in zip(z["img"][m], z["cat"][m], z["bbox"][m], z["score"][m]):
            out.append({"image_id": int(i), "category_id": int(c), "bbox": [float(x) for x in b], "score": float(s)})
    if set(I) - have: raise RuntimeError(f"{ds} {det} {op}: {len(set(I) - have)} images without detections")
    return out


def build(det, dss):
    t0 = time.time(); chk = {}
    for ds in dss:
        G = G_of(ds); I = list(rt_ids(ds)); D = Data(ds); inv = {p: k for k, p in enumerate(D.rt)}
        Wb = np.vstack([np.ones(len(I))] + [np.bincount([inv[p] for p in s], minlength=len(I)) for s in D.specs("rtboot|1500")]).astype(float)
        for op in ["orig"] + OPS_CORE:
            ff = E3F / ds / det / f"{op}.npz"; fm = E3M / ds / det / f"{op}.npz"
            if fm.exists(): continue
            Dl = dets(ds, det, op, I); z = build_generic(G, Dl, I); ff.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(ff, **z)
            if op in ("orig", "jpeg_3"):
                pm, pw, _, _ = pycoco_map_wap(evaluate_generic(G, Dl, I)); a, b = map_wap(z, np.ones(len(I)))
                chk[f"{ds}|{det}|{op}"] = {"pycoco_map": pm, "fast_map": a, "pycoco_wap": pw, "fast_wap": b, "absdiff": max(abs(pm - a), abs(pw - b))}
            v = 100 * batch_map_wap(z, np.arange(len(I)), Wb if op != "orig" else Wb[:1], THREADS)
            out = {"map|rtfull": v[:1, 0], "wap|rtfull": v[:1, 1]}
            if op != "orig": out.update({"map|rtboot|1500": v[1:, 0], "wap|rtboot|1500": v[1:, 1]})
            fm.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(fm, **out)
            print(ds, det, op, round(float(v[0, 0]), 2), round(time.time() - t0), flush=True)
    f = RES / "e3_f4_checks.json"; prev = load_json(f) if f.exists() else {}; prev.update(chk)
    prev["max_absdiff"] = max(v["absdiff"] for k, v in prev.items() if isinstance(v, dict)) if any(isinstance(v, dict) for v in prev.values()) else None
    save_json(prev, f); print("checks max absdiff", prev["max_absdiff"])


def analyse(newdets):
    R = {"label": "EXPLORATORY F4 (post hoc, not preregistered)", "subset": "rtdetr_l 1,500-image subset per dataset; paired rtboot|1500 draws",
         "rule": "|BD(l)-BD(n)|<=6 AND |BD(rt)-BD(l)|>=8 in both JPEG pairs -> family; |BD(l)-BD(rt)|<=6 (both JPEG pairs) -> capacity; else unresolved",
         "datasets": {}}
    for ds in ("coco", "voc"):
        D = Data(ds); dl = [d for d in newdets if (E3M / ds / d / "avif_7.npz").exists()]
        if not dl: continue
        allm = ["yolo26n", "yolo26s", *dl, "rtdetr_l"]; r = {"orig_map": {}, "bd": {}, "paired": {}}
        for m in allm:
            r["orig_map"][m] = float(D.Q(m, "orig", "map", "rtfull")[0])
            r["bd"][m] = {f"{a}->{b}": D.bd("iso", m, a, b, "map", "rtfull", 0) for a, b in PAIRS}
        bv = {m: {f"{a}->{b}": D.bdv("iso", m, a, b, "map", "rtboot|1500") for a, b in PAIRS} for m in allm}
        for m in [*dl, "rtdetr_l"]:
            for ref in ["yolo26n"] + ([d for d in dl if d != m] if m == "rtdetr_l" else []):
                for p in bv[m]:
                    r["paired"][f"{m}-{ref}|{p}"] = {"diff": r["bd"][m][p] - r["bd"][ref][p], "ci95": ci(bv[m][p] - bv[ref][p])}
        if "yolo26l" in dl:
            L, N_, T = r["bd"]["yolo26l"], r["bd"]["yolo26n"], r["bd"]["rtdetr_l"]; jp = [f"{a}->{b}" for a, b in JPEG_PAIRS]
            fam = all(abs(L[p] - N_[p]) <= 6 and abs(T[p] - L[p]) >= 8 for p in jp); cap = all(abs(L[p] - T[p]) <= 6 for p in jp)
            r["rule"] = {"|l-n|": {p: abs(L[p] - N_[p]) for p in jp}, "|rt-l|": {p: abs(T[p] - L[p]) for p in jp},
                         "verdict": "ung ho hieu ung ho" if fam else ("ung ho hieu ung dung luong" if cap else "chua tach duoc")}
        x = np.array([r["orig_map"][m] for m in allm])
        r["bd_vs_map_spearman"] = {p: float(__import__("scipy.stats").stats.spearmanr(x, [r["bd"][m][p] for m in allm])[0]) for p in r["bd"]["yolo26n"]}
        R["datasets"][ds] = r; print(ds, r["orig_map"], {m: {p: round(v, 2) for p, v in b.items()} for m, b in r["bd"].items()}, r.get("rule"), flush=True)
    save_json(R, RES / "e3_f4.json")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "build": build(a[1], a[2:])
    else: analyse(a[1:])
