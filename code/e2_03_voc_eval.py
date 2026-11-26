"""E step 2 C: COCOeval.evaluate() once per (det, op) on VOC (20 classes, COCO ids) -> fast npz WV/fast/<tag>/<det>/<op>.npz.
build_generic = p1_fastmap.build with the GT object and detections passed in (same code path; p1_fastmap is not modified).
Modes:  regress  - rebuild one COCO point with build_generic and compare with the stored P1 fast npz (|diff| <= 1e-12)
        build    - all VOC (det, op)
        check    - DUNG-V2 (fast vs pycocotools, mAP and wAP, <= 1e-9) and DUNG-V1 (yolo26n orig mAP in [30, 80])
Usage: python e2_03_voc_eval.py regress|build|check"""
import sys, io, time, contextlib
import numpy as np
from pycocotools.cocoeval import COCOeval
from e2_voc_common import *
from e2_fastwap import map_wap, pycoco_map_wap


def evaluate_generic(G, D, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(D) if D else None
        E = COCOeval(G, dt, "bbox")
        E.params.imgIds = list(ids); E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]; E.params.maxDets = [100]
        E.evaluate()
    return E


def build_generic(G, D, ids):
    E = evaluate_generic(G, D, ids); pos = {i: n for n, i in enumerate(ids)}
    pe = E._paramsEval; K = len(pe.catIds); I = len(pe.imgIds)
    img, cat, sc, tp, fp = [], [], [], [], []
    npig = np.zeros((len(ids), K), np.int32)
    for k in range(K):
        for j in range(I):
            e = E.evalImgs[k * I + j]
            if e is None: continue
            p = pos[pe.imgIds[j]]
            npig[p, k] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            s = np.asarray(e["dtScores"][:100])
            if s.size == 0: continue
            m = np.asarray(e["dtMatches"])[:, :100] > 0; ig = np.asarray(e["dtIgnore"])[:, :100].astype(bool)
            img.append(np.full(s.size, p)); cat.append(np.full(s.size, k)); sc.append(s)
            tp.append(m & ~ig); fp.append(~m & ~ig)
    img = np.concatenate(img); cat = np.concatenate(cat); sc = np.concatenate(sc)
    tp = np.concatenate(tp, 1); fp = np.concatenate(fp, 1)
    rank = np.argsort(np.argsort(np.array(ids)))
    order = np.lexsort((rank[img], -sc, cat))
    return {"img": img[order].astype(np.int32), "cat": cat[order].astype(np.int32), "tp": tp[:, order], "fp": fp[:, order],
            "npig": npig, "recthrs": np.asarray(pe.recThrs), "ids": np.array(ids)}


def voc_dets(det, op, ids):
    keep = set(c["id"] for c in build_gt()["categories"]); ids_s = set(ids); have = set(); out = []
    for f in sorted((WV / "det" / det / op).glob("[0-9][0-9][0-9].npz")):
        z = np.load(f); have |= set(int(i) for i in z["imgs"])
        m = np.isin(z["img"], list(ids_s)) & np.isin(z["cat"], list(keep))
        for i, c, b, s in zip(z["img"][m], z["cat"][m], z["bbox"][m], z["score"][m]):
            out.append({"image_id": int(i), "category_id": int(c), "bbox": [float(x) for x in b], "score": float(s)})
    if ids_s - have: raise RuntimeError(f"{det} {op}: {len(ids_s - have)} images without detections")
    return out


def tag_ids(det):
    return ("rt", rt_subset()) if det == "rtdetr_l" else ("full", ids_all())


def build_voc(G, det, op):
    tag, ids = tag_ids(det); f = WV / "fast" / tag / det / f"{op}.npz"
    if f.exists(): z = np.load(f); return {k: z[k] for k in z.files}
    z = build_generic(G, voc_dets(det, op, ids), ids); f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **z); return z


def main():
    mode = sys.argv[1]; t0 = time.time()
    if mode == "regress":
        import p1_fastmap as pf
        from p1_common import selection
        ids = selection()["ids"]; zn = build_generic(pf.gt(), pf.load_dets("yolo26n", "jpeg_3", ids), ids)
        zo = dict(np.load(pf.FASTF / "full" / "yolo26n" / "jpeg_3.npz"))
        a = pf.mapw_fast(zn, np.ones(len(ids))); b = pf.mapw_fast(zo, np.ones(len(ids)))
        same = all(np.array_equal(zn[k], zo[k]) for k in ("img", "cat", "tp", "fp", "npig", "recthrs", "ids"))
        r = {"point": "COCO yolo26n jpeg_3 full", "map_new": a, "map_stored": b, "absdiff": abs(a - b), "arrays_identical": same,
             "pass": bool(abs(a - b) <= 1e-12)}
        save_json(r, RES / "e2_regress_coco.json"); print(r)
    elif mode == "build":
        G = gt_coco()
        for det, op, _ in det_plan():
            z = build_voc(G, det, op); print(det, op, int(z["cat"].size), round(time.time() - t0), flush=True)
    else:
        G = gt_coco(); res = []
        for det, op in [("yolo26n", "orig"), ("yolo26n", "jpeg_3"), ("yolo26n", "hevc_2"), ("yolo26s", "avif_5"), ("rtdetr_l", "webp_4")]:
            tag, ids = tag_ids(det); z = build_voc(G, det, op)
            pm, pw, _, _ = pycoco_map_wap(evaluate_generic(G, voc_dets(det, op, ids), ids)); fm, fw = map_wap(z, np.ones(len(ids)))
            res.append({"det": det, "op": op, "set": tag, "pycoco_map": pm, "fast_map": fm, "pycoco_wap": pw, "fast_wap": fw,
                        "absdiff": max(abs(pm - fm), abs(pw - fw))}); print(res[-1], flush=True)
        v1 = 100 * res[0]["pycoco_map"]
        out = {"checks": res, "max_absdiff": max(r["absdiff"] for r in res), "DUNG_V2_pass": bool(max(r["absdiff"] for r in res) <= 1e-9),
               "yolo26n_orig_map_x100": v1, "DUNG_V1_pass": bool(30 <= v1 <= 80),
               "n_images": len(ids_all()), "n_gt": len(build_gt()["annotations"]),
               "n_gt_difficult": int(sum(a["iscrowd"] for a in build_gt()["annotations"]))}
        save_json(out, RES / "e2_voc_checks.json"); print({k: v for k, v in out.items() if k != "checks"})


if __name__ == "__main__":
    main()
