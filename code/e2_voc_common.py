"""E step 2 (confirmatory, PASCAL VOC 2007 test): shared definitions (prereg/PREREG_H44_E.md).
Reuses P1 code without changing it: encoder settings (p1_01_encode.encode, levels_all), detector settings (p1_02_detect),
fast evaluation (p1_fastmap._mapw_sel via e2_fastwap). VOC data and derived files live outside OneDrive."""
import io, json, contextlib
import xml.etree.ElementTree as ET
import numpy as np
from h44_cfg import THS, ANN, WEIGHTS, DETECTORS, IMGSZ, DET_CONF_AP, RES, PROJ, load_json, save_json

SEED_E = 20261124
SEED_BOOT, SEED_RT = SEED_E + 200, SEED_E + 210
VOC = THS / "VOC2007" / "VOCdevkit" / "VOC2007"
WV = THS / "H44_voc"                     # derived: enc/, det/, fast/, maps/
GT_JSON = WV / "voc2007_test_coco.json"
CORE = ["jpeg", "webp", "avif"]
OPS_CORE = [f"{c}_{l}" for c in CORE for l in range(8)]
OPS_HEVC = [f"hevc_{l}" for l in range(8)]
EXT = {"jpeg": "jpeg", "webp": "webp", "avif": "avif", "hevc": "heic"}
N_RT = 1500
NS = [166, 500, 1000, 2500]               # bootstrap sizes reported; full-size bootstrap (n = N) added for "smallest n" and K3 CI
NB = 1000
VOC2COCO = {"aeroplane": "airplane", "bicycle": "bicycle", "bird": "bird", "boat": "boat", "bottle": "bottle", "bus": "bus",
            "car": "car", "cat": "cat", "chair": "chair", "cow": "cow", "diningtable": "dining table", "dog": "dog",
            "horse": "horse", "motorbike": "motorcycle", "person": "person", "pottedplant": "potted plant", "sheep": "sheep",
            "sofa": "couch", "train": "train", "tvmonitor": "tv"}


def coco_cats():
    """COCO categories (id, name) from the COCO val2017 annotation file; detector class index k -> sorted(cat ids)[k]."""
    with open(ANN, encoding="utf8") as f:
        cats = json.load(f)["categories"]
    return sorted((c["id"], c["name"]) for c in cats)


def class_map():
    name2id = {n: i for i, n in coco_cats()}
    return {v: (c, name2id[c]) for v, c in VOC2COCO.items()}


def stems():
    return [s.strip() for s in (VOC / "ImageSets" / "Main" / "test.txt").read_text().split() if s.strip()]


def build_gt():
    """VOC xml -> COCO-format json. bbox [xmin-1, ymin-1, xmax-xmin+1, ymax-ymin+1]; difficult=1 -> iscrowd=1."""
    if GT_JSON.exists(): return load_json(GT_JSON)
    cm = class_map(); images, anns = [], []; aid = 1
    for s in stems():
        r = ET.parse(VOC / "Annotations" / f"{s}.xml").getroot(); sz = r.find("size")
        images.append({"id": int(s), "file_name": f"{s}.jpg", "width": int(sz.find("width").text), "height": int(sz.find("height").text)})
        for o in r.findall("object"):
            b = o.find("bndbox"); x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            w, h = x2 - x1 + 1, y2 - y1 + 1; d = o.find("difficult"); dif = int(d.text) if d is not None else 0
            anns.append({"id": aid, "image_id": int(s), "category_id": cm[o.find("name").text][1], "bbox": [x1 - 1, y1 - 1, w, h],
                         "area": w * h, "iscrowd": int(dif == 1), "voc_difficult": dif}); aid += 1
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": cid, "name": cn, "voc_name": v} for v, (cn, cid) in sorted(cm.items(), key=lambda t: t[1][1])]}
    save_json(gt, GT_JSON); return gt


def gt_coco():
    from pycocotools.coco import COCO as COCOapi
    build_gt()
    with contextlib.redirect_stdout(io.StringIO()):
        return COCOapi(str(GT_JSON))


def ids_all():
    return sorted(int(s) for s in stems())


def rt_subset():
    ids = ids_all()
    return sorted(int(ids[k]) for k in np.random.default_rng(SEED_RT).choice(len(ids), N_RT, replace=False))


def img_path(i, op):
    if op == "orig": return VOC / "JPEGImages" / f"{i:06d}.jpg"
    c = op.split("_")[0]; return WV / "enc" / op / f"{i:06d}.{EXT[c]}"


def det_plan():
    ids = ids_all(); rt = rt_subset()
    return [("yolo26n", o, ids) for o in ["orig"] + OPS_CORE + OPS_HEVC] + \
           [("yolo26s", o, ids) for o in ["orig"] + OPS_CORE] + \
           [("rtdetr_l", o, rt) for o in ["orig"] + OPS_CORE]


def ops_of(det):
    return ["orig"] + OPS_CORE + (OPS_HEVC if det == "yolo26n" else [])
