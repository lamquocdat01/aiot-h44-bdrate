"""P1 F1 step 2: detections (same settings as S1a: imgsz 640, conf 0.001, max_det 300, CPU FP32).
Pilot images reuse S1a/S1a2 detections (repro check p1_00_repro.py: identical on 20 images).
Jobs = (det, op, chunk of <=CH images); output WORK/p1/det/<det>/<op>/<k>.npz (resumable).
Usage: python p1_02_detect.py <nproc> <threads_per_proc> [det ...]"""
import sys, time
import numpy as np
from multiprocessing import Pool
from p1_common import *

CH = 250
_M = {}


def init(thr):
    import torch; torch.set_num_threads(thr)
    import pillow_avif, pillow_heif  # noqa
    pillow_heif.register_heif_opener()


def model(det):
    if det not in _M:
        from ultralytics import YOLO, RTDETR
        _M.clear(); _M[det] = (RTDETR if det.startswith("rtdetr") else YOLO)(str(WEIGHTS / DETECTORS[det]))
    return _M[det]


def run(job):
    det, op, k, imgs = job
    from PIL import Image
    fo = P1 / "det" / det / op / f"{k:03d}.npz"
    if fo.exists(): return (det, op, k, 0, 0.0)
    m = model(det); catids = CATIDS
    I, C, BB, S, T = [], [], [], [], []
    for i in imgs:
        im = Image.open(img_path(i, op)).convert("RGB")
        t = time.perf_counter(); r = m.predict(im, imgsz=IMGSZ, conf=DET_CONF_AP, max_det=300, verbose=False, device="cpu")[0]
        T.append(time.perf_counter() - t); b = r.boxes
        xy = b.xyxy.numpy().astype(np.float64); n = len(xy)
        I.append(np.full(n, i, np.int64)); C.append(np.asarray(catids)[b.cls.numpy().astype(int)]); S.append(b.conf.numpy().astype(np.float64))
        BB.append(np.c_[xy[:, :2], xy[:, 2:] - xy[:, :2]])
    fo.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fo, img=np.concatenate(I), cat=np.concatenate(C), bbox=np.concatenate(BB), score=np.concatenate(S),
                        imgs=np.array(imgs), t=np.array(T))
    return (det, op, k, len(imgs), float(np.mean(T)))


CATIDS = None


def main():
    global CATIDS
    nproc, thr = int(sys.argv[1]), int(sys.argv[2]); dets = sys.argv[3:]
    pilot = set(load_json(RES / "s1a_selection.json")["pilot"])
    jobs = []
    for det, op, ids in det_plan():
        if dets and det not in dets: continue
        new = [i for i in ids if i not in pilot] if not op.startswith("hevc") else list(ids)
        for k in range(0, len(new), CH): jobs.append((det, op, k // CH, new[k:k + CH]))
    print("jobs", len(jobs), flush=True); t0 = time.time()
    with Pool(nproc, initializer=init, initargs=(thr,)) as pool:
        for j, r in enumerate(pool.imap_unordered(run, jobs)):
            print(j, r, round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)


G = coco(); CATIDS = sorted(G.getCatIds())
if __name__ == "__main__":
    main()
