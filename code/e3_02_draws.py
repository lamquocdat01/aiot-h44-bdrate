"""S2FIX F2/F3 (post hoc): extra mAP/wAP draws not stored before. Output WORK/p1/e3/maps/<ds>/<det>/<op>.npz (keys map|<key>, wap|<key>).
COCO: P1 bootstrap draws (seed 20261123+30) boot|500 and boot|4952 (wAP; mAP recomputed and checked against stored), new boot|50
(seed 20261125+310); yolo26n/s on the rtdetr_l subset with P1 rtdetr_l draws boot|1500 (seed +35) and rtfull.
VOC: new boot|50, boot|100 (seed 20261125+310).  Usage: python e3_02_draws.py"""
import time
import numpy as np
from e3_common import *

E3M = P1 / "e3" / "maps"


def new_boot(n, N=4952):
    r = np.random.default_rng([SEED_F + 310, n]); return [r.integers(0, N, n) for _ in range(1000)]


def save(ds, det, op, out):
    f = E3M / ds / det / f"{op}.npz"; f.parent.mkdir(parents=True, exist_ok=True)
    old = dict(np.load(f)) if f.exists() else {}; old.update(out); np.savez_compressed(f, **old)


def done(ds, det, op, key):
    f = E3M / ds / det / f"{op}.npz"; return f.exists() and f"map|{key}" in np.load(f).files


def run_full_batch(z, specs, N):
    return 100 * batch_map_wap(z, np.arange(N), counts(specs, N), THREADS)


def main():
    t0 = time.time(); N = 4952; chk = {}
    from p1_04_draws import draws
    for det in ("yolo26n", "yolo26s"):
        D = draws("rest", det); ops = OPS_CORE + (OPS_HEVC if det == "yolo26n" else [])
        rtpos = np.array([{i: k for k, i in enumerate(ids("coco"))}[i] for i in rt_ids("coco")])
        Drt = [p for p, _ in draws("rest", "rtdetr_l")["boot|1500"]]
        for o in ops:
            z = fast("coco", det, o); st = np.load(P1 / "maps" / "rest" / det / f"{o}.npz"); out = {}
            for key in (["boot|500"] if o in OPS_CORE else []) + ["boot|4952"]:
                if done("coco", det, o, key): continue
                v = run_full_batch(z, [p for p, _ in D[key]], N); out[f"map|{key}"] = v[:, 0]; out[f"wap|{key}"] = v[:, 1]
                chk[f"{det}|{o}|{key}"] = float(np.max(np.abs(v[:, 0] - st[key])))
            if o in OPS_CORE and not done("coco", det, o, "boot|50"):
                v = 100 * rows_map_wap(z, new_boot(50), THREADS); out["map|boot|50"] = v[:, 0]; out["wap|boot|50"] = v[:, 1]
            if o in OPS_CORE and not done("coco", det, o, "rtboot|1500"):
                W = np.vstack([np.ones(len(rtpos))] + [np.bincount(p, minlength=len(rtpos)) for p in Drt]).astype(float)
                v = 100 * batch_map_wap(z, rtpos, W, THREADS)
                out["map|rtfull"] = v[:1, 0]; out["wap|rtfull"] = v[:1, 1]; out["map|rtboot|1500"] = v[1:, 0]; out["wap|rtboot|1500"] = v[1:, 1]
            if out: save("coco", det, o, out)
            print("coco", det, o, round(time.time() - t0), flush=True)
    for det in ("yolo26n", "yolo26s"):
        for o in OPS_CORE:
            out = {}; z = fast("voc", det, o)
            for n in (50, 100):
                if done("voc", det, o, f"boot|{n}"): continue
                v = 100 * rows_map_wap(z, new_boot(n), THREADS); out[f"map|boot|{n}"] = v[:, 0]; out[f"wap|boot|{n}"] = v[:, 1]
            if out: save("voc", det, o, out)
            print("voc", det, o, round(time.time() - t0), flush=True)
    prev = load_json(RES / "e3_draws_check.json") if (RES / "e3_draws_check.json").exists() else {}
    prev.update(chk); prev["max_abs_diff_map_vs_stored_x100"] = max(v for k, v in prev.items() if "|" in k) if any("|" in k for k in prev) else None
    save_json(prev, RES / "e3_draws_check.json"); print("max diff vs stored (x100)", prev["max_abs_diff_map_vs_stored_x100"], round(time.time() - t0))


if __name__ == "__main__":
    main()
