"""E step 2 C/D: mAP and wAP (x100) for every bootstrap draw on VOC. Paired: the same index draws for every (det, op).
Full set (yolo26n/s, N = 4952): boot|n, n in {166, 500, 1000, 2500, N}, 1000 draws each, stream default_rng([SEED+200, n]).
rtdetr_l subset (1500, seed +210): rtboot|1500, 1000 draws, stream default_rng([SEED+200, 1500, 1]) over subset positions;
yolo26n/s evaluated on the same subset draws (paired with rtdetr_l).
Output WV/maps/<det>/<op>.npz with keys map|<key>, wap|<key> (key: full, boot|n, rtfull, rtboot|1500).
Usage: python e2_04_voc_draws.py <nproc>"""
import sys, time
import numpy as np
from multiprocessing import Pool
from e2_voc_common import *
from e2_fastwap import map_wap


def specs(N):
    """dict key -> list of position arrays (into ids_all()); kind 'boot' (counts) except full/rtfull (0/1)."""
    ids = ids_all(); pos = {i: k for k, i in enumerate(ids)}; rt = np.array([pos[i] for i in rt_subset()])
    out = {"full": [np.arange(N)], "rtfull": [rt]}
    for n in NS + [N]:
        r = np.random.default_rng([SEED_BOOT, n]); out[f"boot|{n}"] = [r.integers(0, N, n) for _ in range(NB)]
    r = np.random.default_rng([SEED_BOOT, N_RT, 1]); out[f"rtboot|{N_RT}"] = [rt[r.integers(0, N_RT, N_RT)] for _ in range(NB)]
    return out, rt


def job(a):
    det, op = a; fo = WV / "maps" / det / f"{op}.npz"
    if fo.exists(): return (det, op, "skip")
    t = time.time(); N = len(ids_all()); S, rt = specs(N)
    tag = "rt" if det == "rtdetr_l" else "full"; z = dict(np.load(WV / "fast" / tag / det / f"{op}.npz"))
    if det == "rtdetr_l":
        keys = ["rtfull", f"rtboot|{N_RT}"]; ipos = {p: k for k, p in enumerate(rt)}; M = len(rt)
        to_w = lambda p: np.bincount([ipos[x] for x in p], minlength=M).astype(float)
    else:
        keys = list(S); to_w = lambda p: np.bincount(p, minlength=N).astype(float)
    out = {}
    for k in keys:
        v = np.array([map_wap(z, to_w(p)) for p in S[k]]) * 100
        out[f"map|{k}"] = v[:, 0]; out[f"wap|{k}"] = v[:, 1]
    fo.parent.mkdir(parents=True, exist_ok=True); tmp = fo.with_name(fo.stem + "_tmp.npz"); np.savez_compressed(tmp, **out); tmp.replace(fo)
    return (det, op, round(time.time() - t))


if __name__ == "__main__":
    nproc = int(sys.argv[1]); J = [(d, o) for d, o, _ in det_plan()]; t0 = time.time(); print("jobs", len(J), flush=True)
    with Pool(nproc) as pool:
        for r in pool.imap_unordered(job, J, chunksize=1): print(r, round(time.time() - t0), flush=True)
    print("done", round(time.time() - t0), flush=True)
