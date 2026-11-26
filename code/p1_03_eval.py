"""P1 F1 step 3: COCOeval.evaluate() once per (det, op) on its image set -> fast npz (WORK/p1/fast/<tag>/<det>/<op>.npz).
Cross-check (DUNG-1): pycocotools full accumulate/summarize mAP vs fast mAP (all weights 1) on >= 3 points of the full set.
Usage: python p1_03_eval.py build <nproc> [det ...] | check"""
import sys, time
import numpy as np
from multiprocessing import Pool
from p1_common import *
from p1_fastmap import build, mapw_fast, pycoco_map


def plan():
    s = selection()
    out = []
    for det, op, ids in det_plan():
        tag = "rt" if det == "rtdetr_l" else "full"
        out.append((det, op, ids, tag))
    return out


def job(a):
    det, op, ids, tag = a; t = time.time()
    try:
        z = build(det, op, ids, tag); return (det, op, tag, int(z["cat"].size), round(time.time() - t))
    except Exception as e:
        return (det, op, tag, "ERR " + str(e)[:200], 0)


def main():
    if sys.argv[1] == "build":
        nproc = int(sys.argv[2]); dets = sys.argv[3:]
        jobs = [j for j in plan() if not dets or j[0] in dets]
        with Pool(nproc) as pool:
            for r in pool.imap_unordered(job, jobs, chunksize=1): print(r, flush=True)
    else:
        s = selection(); ids = s["ids"]; res = []
        for det, op in [("yolo26n", "orig"), ("yolo26n", "jpeg_3"), ("yolo26s", "avif_5"), ("yolo26n", "hevc_2")]:
            z = build(det, op, ids, "full"); a = pycoco_map(det, op, ids); b = mapw_fast(z, np.ones(len(ids)))
            res.append({"det": det, "op": op, "pycocotools": a, "fast": b, "absdiff": abs(a - b)}); print(res[-1], flush=True)
        rt = s["rtdetr_subset"]; z = build("rtdetr_l", "webp_4", rt, "rt"); a = pycoco_map("rtdetr_l", "webp_4", rt); b = mapw_fast(z, np.ones(len(rt)))
        res.append({"det": "rtdetr_l", "op": "webp_4", "set": "rt1500", "pycocotools": a, "fast": b, "absdiff": abs(a - b)}); print(res[-1])
        ok = max(r["absdiff"] for r in res) <= 1e-6
        save_json({"checks": res, "max_absdiff": max(r["absdiff"] for r in res), "DUNG1_pass": ok}, RES / "p1_fastmap_check.json")
        print("DUNG-1 pass" if ok else "DUNG-1 FAIL")


if __name__ == "__main__":
    main()
