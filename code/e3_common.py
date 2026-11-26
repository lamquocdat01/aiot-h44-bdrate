"""S2FIX shared helpers (post hoc / exploratory; definitions S2/E_S2FIX_H44.md section 0). Two datasets: 'coco' (P1), 'voc' (E-S2)."""
import warnings
import numpy as np
from p1_bd import bd_rate, bd_iso, computable_std
from e3_fastboot import batch_map_wap, rows_map_wap
import e2_voc_common as V
from h44_cfg import RES, load_json, save_json
from p1_common import P1, selection

warnings.simplefilter("ignore")
SEED_F = 20261125
CORE = ["jpeg", "webp", "avif"]
OPS_CORE = [f"{c}_{l}" for c in CORE for l in range(8)]
OPS_HEVC = [f"hevc_{l}" for l in range(8)]
PAIRS = [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]
GRIDS = {"L4": list(range(4)), "L5": list(range(5)), "L6": list(range(6)), "L7": list(range(7)), "L8": list(range(8))}
THREADS = 6


def ids(ds): return selection()["ids"] if ds == "coco" else V.ids_all()


def rt_ids(ds): return selection()["rtdetr_subset"] if ds == "coco" else V.rt_subset()


def fast(ds, det, op, tag="full"):
    base = P1 / "fast" if ds == "coco" else V.WV / "fast"
    z = np.load(base / tag / det / f"{op}.npz"); return {k: z[k] for k in z.files}


def bpp_matrix(ds, ops=OPS_CORE + OPS_HEVC):
    """(N, len(ops)) per-image bpp in ids(ds) order"""
    I = ids(ds); pos = {i: k for k, i in enumerate(I)}; oi = {o: j for j, o in enumerate(ops)}
    z = np.load(P1 / "bpp.npz") if ds == "coco" else np.load(V.WV / "bpp.npz"); B = np.zeros((len(I), len(ops)))
    for i, o, b in zip(z["img"], z["op"], z["bpp"]):
        o = str(o)
        if o in oi and int(i) in pos: B[pos[int(i)], oi[o]] = b
    assert (B > 0).all(); return B


def counts(specs, N):
    """list of index arrays (positions, with repetition) -> (B, N) count matrix"""
    W = np.zeros((len(specs), N))
    for b, p in enumerate(specs): W[b] = np.bincount(p, minlength=N)
    return W


def maps_spec(ds, det, op, specs, tag="full", z=None):
    """mAP, wAP x100 for each spec (index array over the npz image positions; repetitions = weights)"""
    z = fast(ds, det, op, tag) if z is None else z
    return 100 * rows_map_wap(z, specs, THREADS)


def bd(m, r1, q1, r2, q2):
    if m == "pchip": return bd_rate(r1, q1, r2, q2) if computable_std(q1, q2) else np.nan
    if m == "akima":
        from e2_05_confirm import bd_akima
        return bd_akima(r1, q1, r2, q2)
    return bd_iso(r1, q1, r2, q2)


def hw(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return float((np.percentile(v, 97.5) - np.percentile(v, 2.5)) / 2) if v.size >= 20 else None


def ci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size >= 20 else None


def wilson(k, n, z=1.959964):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]
