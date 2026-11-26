"""E-FINISH: collect full-set rate-accuracy curves (mean bpp, mAP x100) for the supplement into results/e4_curves.json,
so that tables are generated from json only. Reads stored per-op evaluations (no inference). Usage: python e4_00_collect.py"""
from e3_common import *
from e3_data import Data, BOPS


def main():
    R = {"label": "full-set curves (4,952 images per dataset); 1,500-image subset for every detector (rtfull)", "full": {}, "rtfull": {}}
    for ds in ("coco", "voc"):
        D = Data(ds); bf = dict(zip(BOPS, D.BP("full")[0])); br = dict(zip(BOPS, D.BP("rtfull")[0]))
        for det in ("yolo26n", "yolo26s"):
            R["full"][f"{ds}|{det}"] = {"orig": float(D.Q(det, "orig", "map", "full")[0]),
                                        **{c: {"bpp": [float(bf[f"{c}_{l}"]) for l in range(8)], "map": [float(D.Q(det, f"{c}_{l}", "map", "full")[0]) for l in range(8)]} for c in CORE}}
        for det in ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l"):
            R["rtfull"][f"{ds}|{det}"] = {"orig": float(D.Q(det, "orig", "map", "rtfull")[0]),
                                          **{c: {"bpp": [float(br[f"{c}_{l}"]) for l in range(8)], "map": [float(D.Q(det, f"{c}_{l}", "map", "rtfull")[0]) for l in range(8)]} for c in CORE}}
    save_json(R, RES / "e4_curves.json"); print({k: round(v["orig"], 2) for k, v in R["full"].items()})


if __name__ == "__main__":
    main()
