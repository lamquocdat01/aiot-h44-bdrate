"""P1 fast weighted mAP@[.5:.95] (area all, maxDet 100) on arbitrary image sets (numba).
build(det, op, ids): pycocotools COCOeval.evaluate() once on all ids -> per-detection match flags (cached npz).
mapw(z, w): mAP for image weights w (0/1 subsets or bootstrap counts); same weighted re-accumulation as h44_fastmap.wmap
(validated against pycocotools on the pilot in S1a; re-validated on the full set in p1_03_eval.py)."""
import io, contextlib
from pycocotools.cocoeval import COCOeval
import numpy as np
from numba import njit
from p1_common import *

FASTF = P1 / "fast"
_GT = None


def gt():
    global _GT
    if _GT is None: _GT = coco()
    return _GT


def load_dets(det, op, ids):
    """detections for ids: pilot images from S1a/S1a2 JSON, others from P1 parts."""
    ids_s = set(ids); pilot = set(load_json(RES / "s1a_selection.json")["pilot"]); out = []
    if not op.startswith("hevc") and (ids_s & pilot):
        for d in load_json(DET_DIR / det / f"{op}.json")["dets"]:
            if d["image_id"] in ids_s: out.append(d)
    pdir = P1 / "det" / det / op
    have = set(i for i in ids_s & pilot) if not op.startswith("hevc") else set()
    for f in sorted(pdir.glob("*.npz")) if pdir.exists() else []:
        z = np.load(f)
        for i in z["imgs"]: have.add(int(i))
        m = np.isin(z["img"], list(ids_s))
        for i, c, b, s in zip(z["img"][m], z["cat"][m], z["bbox"][m], z["score"][m]):
            out.append({"image_id": int(i), "category_id": int(c), "bbox": [float(x) for x in b], "score": float(s)})
    missing = ids_s - have
    if missing: raise RuntimeError(f"{det} {op}: {len(missing)} images without detections")
    return out


def evaluate(det, op, ids, D=None):
    G = gt(); D = load_dets(det, op, ids) if D is None else D
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(D) if D else None
        E = COCOeval(G, dt, "bbox")
        E.params.imgIds = list(ids); E.params.areaRng = [[0, 1e10]]; E.params.areaRngLbl = ["all"]; E.params.maxDets = [100]
        E.evaluate()
    return E


def build(det, op, ids, tag):
    f = FASTF / tag / det / f"{op}.npz"
    if f.exists():
        z = np.load(f); return {k: z[k] for k in z.files}
    E = evaluate(det, op, ids)
    pos = {i: n for n, i in enumerate(ids)}
    pe = E._paramsEval; K = len(pe.catIds); I = len(pe.imgIds)
    img, cat, sc, tp, fp = [], [], [], [], []
    npig = np.zeros((len(ids), K), np.int32)
    for k in range(K):
        for j in range(I):
            e = E.evalImgs[k * I + j]
            if e is None: continue
            p = pos[pe.imgIds[j]]
            npig[p, k] = int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            s = np.asarray(e["dtScores"][:100])
            if s.size == 0: continue
            m = np.asarray(e["dtMatches"])[:, :100] > 0; ig = np.asarray(e["dtIgnore"])[:, :100].astype(bool)
            img.append(np.full(s.size, p)); cat.append(np.full(s.size, k)); sc.append(s)
            tp.append(m & ~ig); fp.append(~m & ~ig)
    img = np.concatenate(img); cat = np.concatenate(cat); sc = np.concatenate(sc)
    tp = np.concatenate(tp, 1); fp = np.concatenate(fp, 1)
    rank = np.argsort(np.argsort(np.array(ids)))
    order = np.lexsort((rank[img], -sc, cat))
    out = {"img": img[order].astype(np.int32), "cat": cat[order].astype(np.int32), "tp": tp[:, order], "fp": fp[:, order],
           "npig": npig, "recthrs": np.asarray(pe.recThrs), "ids": np.array(ids)}
    f.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(f, **out)
    return out


def pycoco_map(det, op, ids):
    E = evaluate(det, op, ids)
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate()
    # identical to pycocotools _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100); summarize() itself needs 3 maxDets
    p = E.params; a = p.areaRngLbl.index("all"); m = p.maxDets.index(100)
    s = E.eval["precision"][:, :, :, a, m]
    return float(np.mean(s[s > -1])) if (s > -1).any() else -1.0


@njit(cache=True)
def _mapw(cat, img, tp, fp, npig, w, R):
    K = npig.shape[1]; T = tp.shape[0]; D = cat.shape[0]; NR = R.shape[0]
    npw = np.zeros(K)
    for i in range(npig.shape[0]):
        if w[i] != 0:
            for k in range(K): npw[k] += w[i] * npig[i, k]
    starts = np.searchsorted(cat, np.arange(K), "left"); ends = np.searchsorted(cat, np.arange(K), "right")
    rc = np.empty(D); pr = np.empty(D); tot = 0.0; cnt = 0
    for k in range(K):
        if npw[k] <= 0: continue
        for t in range(T):
            ctp = 0.0; cfp = 0.0; m = 0
            for d in range(starts[k], ends[k]):
                wd = w[img[d]]
                if wd == 0: continue
                if tp[t, d]: ctp += wd
                if fp[t, d]: cfp += wd
                rc[m] = ctp / npw[k]; pr[m] = ctp / (ctp + cfp + 2.220446049250313e-16); m += 1
            for i in range(m - 2, -1, -1):
                if pr[i + 1] > pr[i]: pr[i] = pr[i + 1]
            s = 0.0; i = 0
            for r in range(NR):
                while i < m and rc[i] < R[r]: i += 1
                if i < m: s += pr[i]
            tot += s; cnt += NR
    return tot / cnt if cnt > 0 else np.nan


def mapw(z, w):
    return _mapw(z["cat"], z["img"], z["tp"], z["fp"], z["npig"], np.asarray(w, np.float64), z["recthrs"].astype(np.float64))


@njit(cache=True)
def _mapw_sel(sel, wsel, order_img, off, cat, img, tp, fp, npig, wfull, R):
    """same as _mapw but only visits detections of selected images (sel: image positions, wsel: their weights;
    wfull: dense weight vector, zero outside sel)."""
    K = npig.shape[1]; T = tp.shape[0]; NR = R.shape[0]
    npw = np.zeros(K); tot_d = 0
    for a in range(sel.shape[0]):
        p = sel[a]; tot_d += off[p + 1] - off[p]
        for k in range(K): npw[k] += wsel[a] * npig[p, k]
    idx = np.empty(tot_d, np.int64); m = 0
    for a in range(sel.shape[0]):
        p = sel[a]
        for j in range(off[p], off[p + 1]): idx[m] = order_img[j]; m += 1
    idx = np.sort(idx)
    rc = np.empty(tot_d + 1); pr = np.empty(tot_d + 1); tot = 0.0; cnt = 0
    # category segments within idx
    seg_s = np.full(K, -1, np.int64); seg_e = np.zeros(K, np.int64)
    for j in range(tot_d):
        k = cat[idx[j]]
        if seg_s[k] < 0: seg_s[k] = j
        seg_e[k] = j + 1
    for k in range(K):
        if npw[k] <= 0: continue
        for t in range(T):
            ctp = 0.0; cfp = 0.0; m = 0
            if seg_s[k] >= 0:
                for j in range(seg_s[k], seg_e[k]):
                    d = idx[j]; wd = wfull[img[d]]
                    if tp[t, d]: ctp += wd
                    if fp[t, d]: cfp += wd
                    rc[m] = ctp / npw[k]; pr[m] = ctp / (ctp + cfp + 2.220446049250313e-16); m += 1
            for i in range(m - 2, -1, -1):
                if pr[i + 1] > pr[i]: pr[i] = pr[i + 1]
            s = 0.0; i = 0
            for r in range(NR):
                while i < m and rc[i] < R[r]: i += 1
                if i < m: s += pr[i]
            tot += s; cnt += NR
    return tot / cnt if cnt > 0 else np.nan


def prep(z):
    """CSR index of detections by image (positions in the sorted detection order)."""
    if "_oi" not in z:
        oi = np.argsort(z["img"], kind="stable"); cnt = np.bincount(z["img"], minlength=z["npig"].shape[0])
        z["_oi"] = oi.astype(np.int64); z["_off"] = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        z["_R"] = z["recthrs"].astype(np.float64)
    return z


def mapw_fast(z, w):
    """mAP for a weight vector w (len N); visits only images with w>0."""
    prep(z); w = np.asarray(w, np.float64); sel = np.nonzero(w)[0].astype(np.int64)
    return _mapw_sel(sel, w[sel], z["_oi"], z["_off"], z["cat"], z["img"], z["tp"], z["fp"], z["npig"], w, z["_R"])
