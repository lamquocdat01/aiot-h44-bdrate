"""E step 2: per-class AP@[.5:.95] for image weights (extends p1_fastmap, which is not modified).
wAP = sum_k npw_k * AP_k / sum_k npw_k over classes with npw_k > 0, npw_k = weighted number of non-ignored GT objects of class k
(average over object instances instead of over classes; Herglotz et al. TIP 2024, sec. on VCM, citing Fischer et al.).
mAP = plain mean of AP_k over the same classes (identical to p1_fastmap.mapw_fast up to float summation order).
Works on any npz produced by p1_fastmap.build-style code (keys cat, img, tp, fp, npig, recthrs)."""
import io, contextlib
import numpy as np
from numba import njit
from p1_fastmap import prep


@njit(cache=True)
def _apk_sel(sel, wsel, order_img, off, cat, img, tp, fp, npig, wfull, R):
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
    rc = np.empty(tot_d + 1); pr = np.empty(tot_d + 1)
    seg_s = np.full(K, -1, np.int64); seg_e = np.zeros(K, np.int64)
    for j in range(tot_d):
        k = cat[idx[j]]
        if seg_s[k] < 0: seg_s[k] = j
        seg_e[k] = j + 1
    apk = np.full(K, np.nan)
    for k in range(K):
        if npw[k] <= 0: continue
        tot = 0.0
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
            tot += s
        apk[k] = tot / (T * NR)
    return apk, npw


def apk(z, w):
    """per-class AP (nan where no GT weight) and weighted GT counts"""
    prep(z); w = np.asarray(w, np.float64); sel = np.nonzero(w)[0].astype(np.int64)
    return _apk_sel(sel, w[sel], z["_oi"], z["_off"], z["cat"], z["img"], z["tp"], z["fp"], z["npig"], w, z["_R"])


def map_wap(z, w):
    a, n = apk(z, w); ok = n > 0
    return float(np.mean(a[ok])), float(np.sum(n[ok] * a[ok]) / np.sum(n[ok]))


def pycoco_map_wap(E):
    """reference values from a pycocotools COCOeval after evaluate() (areaRng all, maxDets [100]): accumulate, per-class AP,
    GT counts from evalImgs gtIgnore."""
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate()
    s = E.eval["precision"][:, :, :, 0, 0]; K = s.shape[2]; pe = E._paramsEval; I = len(pe.imgIds)
    ap = np.array([np.mean(s[:, :, k][s[:, :, k] > -1]) if (s[:, :, k] > -1).any() else np.nan for k in range(K)])
    n = np.zeros(K)
    for k in range(K):
        for j in range(I):
            e = E.evalImgs[k * I + j]
            if e is not None: n[k] += int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
    ok = (n > 0) & ~np.isnan(ap)
    return float(np.mean(s[s > -1])), float(np.sum(n[ok] * ap[ok]) / np.sum(n[ok])), ap, n
