"""H44 S1a pilot - shared config. Data/derived files live outside OneDrive (THS_DATASETS)."""
import os, json
from pathlib import Path

SEED = 20261122
THS = Path(os.environ.get("THS_DATASETS", r"D:\THS_Datasets"))
COCO = THS / "COCO2017"
ANN = COCO / "annotations" / "instances_val2017.json"
IMG_DIR = COCO / "val2017"
WORK = THS / "H44_pilot"            # derived: compressed images, detections, proxies
ENC_DIR = WORK / "enc"
DET_DIR = WORK / "det"
WEIGHTS = WORK / "weights"
PROJ = Path(__file__).resolve().parents[1]
RES = PROJ / "results"

N_IMG, N_GROUP = 500, 10
N_CAL_Q = 40                         # extra images (disjoint) used only to pick codec quality levels
CODECS = ["jpeg", "webp", "avif"]
BPP_TARGETS = [0.15, 0.3, 0.5, 0.8, 1.3]   # mean-bpp targets for the 5 quality levels per codec
DETECTORS = {"yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt", "rtdetr_l": "rtdetr-l.pt"}
DET_CONF_AP = 0.001                  # low threshold for AP
CONS_CONF, CONS_IOU = 0.25, 0.5      # consistency F1 (vs. detections on original)
IMGSZ = 640


def load_json(p):
    with open(p, encoding="utf8") as f:
        return json.load(f)


def save_json(obj, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=1)


def op_list():
    """operating points: list of (op_id, codec, level_idx); 'orig' is the reference."""
    q = load_json(RES / "s1a_quality_levels.json")["levels"]
    ops = [("orig", "orig", -1)]
    for c in CODECS:
        for i, _ in enumerate(q[c]):
            ops.append((f"{c}_{i}", c, i))
    return ops
