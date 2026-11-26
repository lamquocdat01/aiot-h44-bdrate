"""S2FIX: batched mAP / wAP (same arithmetic as e2_fastwap._apk_sel / p1_fastmap._mapw_sel; validated in e3_00_check.py).
batch_map_wap(z, sample, W): one fixed sample of distinct image positions, W (B, len(sample)) weights.
rows_map_wap(z, specs): one weight vector per row given as index arrays over image positions (with repetition)."""
import numpy as np
from numba import njit, prange, set_num_threads
from p1_fastmap import prep


@njit(cache=True)
def _gather(sample, order_img, off, cat, img, tp, fp, nimg):
    """detections of the sample in global sorted order, copied to contiguous arrays; local image index per detection"""
    loc = np.full(nimg, -1, np.int64); tot_d = 0
    for a in range(sample.shape[0]):
        loc[sample[a]] = a; tot_d += off[sample[a] + 1] - off[sample[a]]
    idx = np.empty(tot_d, np.int64); m = 0
    for a in range(sample.shape[0]):
        p = sample[a]
        for j in range(off[p], off[p + 1]): idx[m] = order_img[j]; m += 1
    idx = np.sort(idx)
    T = tp.shape[0]; ltp = np.empty((T, tot_d), np.bool_); lfp = np.empty((T, tot_d), np.bool_)
    li = np.empty(tot_d, np.int64); lc = np.empty(tot_d, np.int64)
    for j in range(tot_d):
        d = idx[j]; li[j] = loc[img[d]]; lc[j] = cat[d]
        for t in range(T): ltp[t, j] = tp[t, d]; lfp[t, j] = fp[t, d]
    return ltp, lfp, li, lc


@njit(cache=True)
def _one(w, sample, ltp, lfp, li, lc, npig, R):
    """w: weights per sample position. returns (mAP, wAP)"""
    K = npig.shape[1]; T = ltp.shape[0]; NR = R.shape[0]; D = li.shape[0]
    npw = np.zeros(K)
    for a in range(sample.shape[0]):
        if w[a] != 0:
            for k in range(K): npw[k] += w[a] * npig[sample[a], k]
    seg_s = np.full(K, -1, np.int64); seg_e = np.zeros(K, np.int64)
    for j in range(D):
        k = lc[j]
        if seg_s[k] < 0: seg_s[k] = j
        seg_e[k] = j + 1
    rc = np.empty(D + 1); pr = np.empty(D + 1); sm = 0.0; cntk = 0; sw = 0.0; swn = 0.0
    for k in range(K):
        if npw[k] <= 0: continue
        tot = 0.0
        for t in range(T):
            ctp = 0.0; cfp = 0.0; mm = 0
            if seg_s[k] >= 0:
                for j in range(seg_s[k], seg_e[k]):
                    wd = w[li[j]]
                    if wd == 0: continue
                    if ltp[t, j]: ctp += wd
                    if lfp[t, j]: cfp += wd
                    rc[mm] = ctp / npw[k]; pr[mm] = ctp / (ctp + cfp + 2.220446049250313e-16); mm += 1
            for i in range(mm - 2, -1, -1):
                if pr[i + 1] > pr[i]: pr[i] = pr[i + 1]
            s = 0.0; i = 0
            for r in range(NR):
                while i < mm and rc[i] < R[r]: i += 1
                if i < mm: s += pr[i]
            tot += s
        ap = tot / (T * NR); sm += ap; cntk += 1; sw += npw[k] * ap; swn += npw[k]
    return (sm / cntk if cntk > 0 else np.nan), (sw / swn if swn > 0 else np.nan)


@njit(parallel=True, cache=True)
def _batch(sample, W, order_img, off, cat, img, tp, fp, npig, R, nimg):
    ltp, lfp, li, lc = _gather(sample, order_img, off, cat, img, tp, fp, nimg)
    B = W.shape[0]; out = np.empty((B, 2))
    for b in prange(B):
        m, wa = _one(W[b], sample, ltp, lfp, li, lc, npig, R); out[b, 0] = m; out[b, 1] = wa
    return out


@njit(parallel=True, cache=True)
def _rows(samp_cat, samp_off, w_cat, order_img, off, cat, img, tp, fp, npig, R, nimg):
    B = samp_off.shape[0] - 1; out = np.empty((B, 2))
    for b in prange(B):
        s = samp_cat[samp_off[b]:samp_off[b + 1]]; w = w_cat[samp_off[b]:samp_off[b + 1]]
        ltp, lfp, li, lc = _gather(s, order_img, off, cat, img, tp, fp, nimg)
        m, wa = _one(w, s, ltp, lfp, li, lc, npig, R); out[b, 0] = m; out[b, 1] = wa
    return out


def batch_map_wap(z, sample, W, threads=6):
    set_num_threads(threads); prep(z)
    return _batch(np.asarray(sample, np.int64), np.ascontiguousarray(W, np.float64), z["_oi"], z["_off"], z["cat"], z["img"],
                  z["tp"], z["fp"], z["npig"], z["_R"], z["npig"].shape[0])


def rows_map_wap(z, specs, threads=6):
    """specs: list of index arrays (positions, repetition = bootstrap counts). returns (B, 2) fractions"""
    set_num_threads(threads); prep(z); S, Wc, o = [], [], [0]
    for p in specs:
        u, c = np.unique(np.asarray(p), return_counts=True); S.append(u); Wc.append(c); o.append(o[-1] + u.size)
    return _rows(np.concatenate(S).astype(np.int64), np.array(o, np.int64), np.concatenate(Wc).astype(np.float64), z["_oi"], z["_off"],
                 z["cat"], z["img"], z["tp"], z["fp"], z["npig"], z["_R"], z["npig"].shape[0])
