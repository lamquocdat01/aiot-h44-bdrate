"""E-FINISH H1 / S3_DRAFT G1 (exploratory; copy of e3_04_detect.py with the third-family model): detections of extra models on the rtdetr_l 1,500-image subset of each dataset (orig + 3 codecs x 8).
Models: yolo26l (Ultralytics, P1 settings: imgsz 640, conf 0.001, max_det 300, CPU FP32);
frcnn_mv3 = torchvision fasterrcnn_mobilenet_v3_large_fpn, COCO_V1 weights, box_score_thresh 0.001, box_detections_per_img 300, default transform.
One process; chunks of 250 resumable: WORK/p1/e3/det/<ds>/<det>/<op>/<k>.npz.
Usage: python e4_01_frcnn.py time | run <det> <ds> [<ds> ...] --threads T --max-hours H"""
import sys, time
import numpy as np
from e3_common import *
import p1_common as PC

E3D = P1 / "e3" / "det"; CH = 250


def path(ds, i, op): return PC.img_path(i, op) if ds == "coco" else V.img_path(i, op)


class Model:
    def __init__(self, det, thr):
        import torch; torch.set_num_threads(thr); self.det = det
        import pillow_avif, pillow_heif  # noqa
        pillow_heif.register_heif_opener()
        if det == "yolo26l":
            from ultralytics import YOLO
            self.m = YOLO(str(V.WEIGHTS / "yolo26l.pt")); self.cat = np.array([i for i, _ in V.coco_cats()])
        else:
            import torchvision
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
            w = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
            self.m = fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=0.001, box_detections_per_img=300).eval(); self.tf = w.transforms()
            self.torch = torch

    def __call__(self, im):
        if self.det == "yolo26l":
            r = self.m.predict(im, imgsz=V.IMGSZ, conf=V.DET_CONF_AP, max_det=300, verbose=False, device="cpu")[0]; b = r.boxes
            return b.xyxy.numpy().astype(np.float64), self.cat[b.cls.numpy().astype(int)], b.conf.numpy().astype(np.float64)
        with self.torch.no_grad():
            o = self.m([self.tf(im)])[0]
        return o["boxes"].numpy().astype(np.float64), o["labels"].numpy().astype(np.int64), o["scores"].numpy().astype(np.float64)


def timing():
    from PIL import Image
    I = rt_ids("coco")[:20]; out = {}
    for det in ("frcnn_mv3",):
        m = Model(det, 4); ims = [Image.open(path("coco", i, "jpeg_3")).convert("RGB") for i in I]; m(ims[0]); t = time.time()
        for im in ims: m(im)
        out[det] = (time.time() - t) / len(ims)
    out["est_hours_per_dataset"] = {d: out[d] * 1500 * 25 / 3600 for d in ("frcnn_mv3",)}
    save_json(out, RES / "e4_g1_timing.json"); print(out)


def run(det, dss, thr, maxh):
    from PIL import Image
    m = Model(det, thr); t0 = time.time()
    for ds in dss:
        I = rt_ids(ds); jobs = [(op, k // CH, I[k:k + CH]) for op in ["orig"] + OPS_CORE for k in range(0, len(I), CH)]
        for n, (op, k, imgs) in enumerate(jobs):
            fo = E3D / ds / det / op / f"{k:03d}.npz"
            if fo.exists(): continue
            if (time.time() - t0) / 3600 > maxh: print("STOP max-hours checkpoint", ds, n, flush=True); return
            Ii, C, BB, S, T = [], [], [], [], []
            for i in imgs:
                im = Image.open(path(ds, i, op)).convert("RGB"); t = time.perf_counter(); xy, c, s = m(im); T.append(time.perf_counter() - t)
                Ii.append(np.full(len(xy), i, np.int64)); C.append(c); S.append(s); BB.append(np.c_[xy[:, :2], xy[:, 2:] - xy[:, :2]])
            fo.parent.mkdir(parents=True, exist_ok=True); tmp = fo.with_name(fo.stem + "_tmp.npz")
            np.savez_compressed(tmp, img=np.concatenate(Ii), cat=np.concatenate(C), bbox=np.concatenate(BB), score=np.concatenate(S), imgs=np.array(imgs), t=np.array(T))
            tmp.replace(fo); print(ds, det, op, k, round(float(np.mean(T)), 3), round(time.time() - t0), flush=True)
    print("done", det, dss, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "time": timing()
    else:
        thr = int(a[a.index("--threads") + 1]); maxh = float(a[a.index("--max-hours") + 1])
        run(a[1], [x for x in a[2:] if x in ("coco", "voc")], thr, maxh)
