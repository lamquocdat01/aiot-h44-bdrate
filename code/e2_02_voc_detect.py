"""E step 2 C: detections on VOC (settings of p1_02_detect: imgsz 640, conf 0.001, max_det 300, CPU FP32; all 80 COCO
classes stored, the 60 non-VOC classes are dropped at evaluation). One process. Chunks of 250 images, resumable:
WV/det/<det>/<op>/<k>.npz.  Usage: python e2_02_voc_detect.py <threads> [det ...] [--max-hours H]"""
import sys, time
import numpy as np
from e2_voc_common import *

CH = 250


def main():
    a = sys.argv[1:]; thr = int(a[0]); maxh = float(a[a.index("--max-hours") + 1]) if "--max-hours" in a else 1e9
    dets = [x for x in a[1:] if x in DETECTORS]
    import torch; torch.set_num_threads(thr)
    import pillow_avif, pillow_heif  # noqa
    pillow_heif.register_heif_opener()
    from PIL import Image
    from ultralytics import YOLO, RTDETR
    catids = np.array([i for i, _ in coco_cats()])
    jobs = [(det, op, k // CH, ids[k:k + CH]) for det, op, ids in det_plan() if not dets or det in dets for k in range(0, len(ids), CH)]
    todo = [j for j in jobs if not (WV / "det" / j[0] / j[1] / f"{j[2]:03d}.npz").exists()]
    print("jobs", len(jobs), "todo", len(todo), flush=True); t0 = time.time(); M = {}
    for n, (det, op, k, imgs) in enumerate(todo):
        if (time.time() - t0) / 3600 > maxh: print("STOP max-hours checkpoint", n, flush=True); break
        if det not in M: M.clear(); M[det] = (RTDETR if det.startswith("rtdetr") else YOLO)(str(WEIGHTS / DETECTORS[det]))
        m = M[det]; I, C, BB, S, T = [], [], [], [], []
        for i in imgs:
            im = Image.open(img_path(i, op)).convert("RGB")
            t = time.perf_counter(); r = m.predict(im, imgsz=IMGSZ, conf=DET_CONF_AP, max_det=300, verbose=False, device="cpu")[0]
            T.append(time.perf_counter() - t); b = r.boxes; xy = b.xyxy.numpy().astype(np.float64)
            I.append(np.full(len(xy), i, np.int64)); C.append(catids[b.cls.numpy().astype(int)]); S.append(b.conf.numpy().astype(np.float64))
            BB.append(np.c_[xy[:, :2], xy[:, 2:] - xy[:, :2]])
        fo = WV / "det" / det / op / f"{k:03d}.npz"; fo.parent.mkdir(parents=True, exist_ok=True); tmp = fo.with_name(fo.stem + "_tmp.npz")
        np.savez_compressed(tmp, img=np.concatenate(I), cat=np.concatenate(C), bbox=np.concatenate(BB), score=np.concatenate(S),
                            imgs=np.array(imgs), t=np.array(T)); tmp.replace(fo)
        print(n, det, op, k, len(imgs), round(float(np.mean(T)), 3), round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
