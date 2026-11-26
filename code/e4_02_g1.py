"""E-FINISH H1 / S3_DRAFT G1 (EXPLORATORY): third detector family = torchvision fasterrcnn_mobilenet_v3_large_fpn (COCO_V1).
Evaluation reuses e3_05_f4eval.build (unchanged; COCOeval once per op, paired rtboot|1500 maps). Reading rule recorded in
prereg/AMENDMENT_LOG_E.md #8 before the run (f = frcnn_mv3; pairs J->W, J->A; per dataset):
  |BD(f)-BD(yolo26n)| <= 6 in both pairs                      -> "RT-DETR-L separates from the four CNN detectors"
  |BD(f)-BD(yolo26n)| >= 8 and CI excludes 0 in both pairs     -> "BD-rate depends on the detector, not only RT-DETR"
  otherwise                                                    -> "unclear"
Usage: python e4_02_g1.py [ds ...]  -> results/e4_g1.json"""
import sys
import numpy as np
from e3_common import *
from e3_data import Data, E3M

F = "frcnn_mv3"; REFS = ["yolo26n", "yolo26s", "yolo26l", "rtdetr_l"]; JP = ["jpeg->webp", "jpeg->avif"]
RULE = {"sep": "RT-DETR-L separates from the four CNN detectors", "dep": "BD-rate depends on the detector, not only RT-DETR", "unc": "unclear"}


def analyse(ds):
    D = Data(ds); r = {"orig_map": {}, "bd": {}, "paired": {}}
    allm = REFS + [F]
    for m in allm:
        r["orig_map"][m] = float(D.Q(m, "orig", "map", "rtfull")[0])
        r["bd"][m] = {f"{a}->{b}": float(D.bd("iso", m, a, b, "map", "rtfull", 0)) for a, b in PAIRS}
    bv = {m: {f"{a}->{b}": D.bdv("iso", m, a, b, "map", "rtboot|1500") for a, b in PAIRS} for m in allm}
    for ref in ("yolo26n", "yolo26l", "rtdetr_l"):
        for p in bv[F]:
            d = bv[F][p] - bv[ref][p]
            r["paired"][f"{F}-{ref}|{p}"] = {"diff": r["bd"][F][p] - r["bd"][ref][p], "ci95": ci(d), "n_nan": int(np.isnan(d).sum())}
    ab = {p: abs(r["paired"][f"{F}-yolo26n|{p}"]["diff"]) for p in JP}
    cie = {p: not (r["paired"][f"{F}-yolo26n|{p}"]["ci95"][0] <= 0 <= r["paired"][f"{F}-yolo26n|{p}"]["ci95"][1]) for p in JP}
    v = "sep" if all(ab[p] <= 6 for p in JP) else ("dep" if all(ab[p] >= 8 and cie[p] for p in JP) else "unc")
    r["rule"] = {"|f-n|": ab, "ci_excludes_0": cie, "verdict_key": v, "verdict": RULE[v]}
    return r


def main():
    dss = sys.argv[1:] or ["voc", "coco"]; f = RES / "e4_g1.json"
    R = load_json(f) if f.exists() else {"label": "EXPLORATORY G1 (not preregistered; both datasets seen). Rule: prereg/AMENDMENT_LOG_E.md #8",
                                         "model": "torchvision fasterrcnn_mobilenet_v3_large_fpn COCO_V1, box_score_thresh 0.001, 300 det/img, default transform",
                                         "subset": "rtdetr_l 1,500-image subset; paired rtboot|1500 draws", "datasets": {}}
    for ds in dss:
        if not (E3M / ds / F / "avif_7.npz").exists(): print("skip", ds); continue
        R["datasets"][ds] = analyse(ds); x = R["datasets"][ds]
        print(ds, {k: round(v, 2) for k, v in x["orig_map"].items()}, {k: round(v["diff"], 2) for k, v in x["paired"].items()}, x["rule"]["verdict"])
    save_json(R, f)


if __name__ == "__main__":
    main()
