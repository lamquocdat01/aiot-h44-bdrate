"""P1 shared definitions: image sets, ops, seeds (prereg/PREREG_H44_P1.md)."""
import io, contextlib
import numpy as np
from h44_cfg import *

SEED_P1 = 20261123
P1 = WORK / "p1"
CORE = ["jpeg", "webp", "avif"]
OPS_CORE = [f"{c}_{l}" for c in CORE for l in range(8)]
OPS_HEVC = [f"hevc_{l}" for l in range(8)]
EXT = {"jpeg": "jpeg", "webp": "webp", "avif": "avif", "hevc": "heic"}


def coco():
    from pycocotools.coco import COCO as COCOapi
    with contextlib.redirect_stdout(io.StringIO()):
        return COCOapi(str(ANN))


def selection():
    f = RES / "p1_selection.json"
    if f.exists(): return load_json(f)
    G = coco(); ids = sorted(i for i in G.getImgIds() if len(G.getAnnIds(imgIds=i, iscrowd=False)) >= 1)
    pilot = load_json(RES / "s1a_selection.json")["pilot"]; ps = set(pilot)
    rest = [i for i in ids if i not in ps]
    extra = sorted(int(x) for x in np.random.default_rng(SEED_P1 + 10).choice(rest, 1000, replace=False))
    perm = np.random.default_rng(SEED_P1 + 40).permutation(len(ids))
    A = sorted(ids[k] for k in perm[:len(ids) // 2]); B = sorted(ids[k] for k in perm[len(ids) // 2:])
    c5 = sorted(ids[k] for k in np.random.default_rng(SEED_P1 + 50).permutation(len(ids))[:1000])
    s = {"ids": ids, "rtdetr_subset": sorted(pilot + extra), "A": A, "B": B, "c5": c5,
         "seeds": {"rtdetr_extra": SEED_P1 + 10, "AB": SEED_P1 + 40, "c5": SEED_P1 + 50}}
    save_json(s, f); return s


def img_path(i, op, G=None):
    if op == "orig": return IMG_DIR / f"{i:012d}.jpg"
    c = op.split("_")[0]; return ENC_DIR / op / f"{i:012d}.{EXT[c]}"


def det_plan():
    s = selection(); ids = s["ids"]
    return [("yolo26n", o, ids) for o in ["orig"] + OPS_CORE + OPS_HEVC] + \
           [("yolo26s", o, ids) for o in ["orig"] + OPS_CORE] + \
           [("rtdetr_l", o, s["rtdetr_subset"]) for o in ["orig"] + OPS_CORE]
