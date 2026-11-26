"""S2FIX: unified access to stored + new draws for both datasets (post hoc). Q(det, op, met, key) -> array (x100);
BP(key) -> (B, 32) mean bpp per draw (ops OPS_CORE + OPS_HEVC). Keys: full, boot|n, rtfull, rtboot|1500."""
import numpy as np
from e3_common import *

E3M = P1 / "e3" / "maps"
BOPS = OPS_CORE + OPS_HEVC


class Data:
    def __init__(self, ds):
        self.ds = ds; self.N = 4952; self._q = {}; self._bp = {}; self.Bm = bpp_matrix(ds, BOPS)
        I = ids(ds); pos = {i: k for k, i in enumerate(I)}; self.rt = np.array([pos[i] for i in rt_ids(ds)])
        self._specs = None

    def specs(self, key):
        if self._specs is None:
            S = {}
            if self.ds == "voc":
                from e2_04_voc_draws import specs as vs
                S0, _ = vs(self.N); S.update(S0)
            else:
                from p1_04_draws import draws
                D = draws("rest", "yolo26n"); S.update({k: [p for p, _ in v] for k, v in D.items() if k.startswith("boot|") or k == "full"})
                S["rtfull"] = [self.rt]; S["rtboot|1500"] = [self.rt[p] for p, _ in draws("rest", "rtdetr_l")["boot|1500"]]
            for n in (50, 100):
                if f"boot|{n}" not in S:
                    r = np.random.default_rng([SEED_F + 310, n]); S[f"boot|{n}"] = [r.integers(0, self.N, n) for _ in range(1000)]
            self._specs = S
        return self._specs[key]

    def BP(self, key):
        if key not in self._bp:
            self._bp[key] = np.array([np.bincount(p, minlength=self.N) @ self.Bm / len(p) for p in self.specs(key)])
        return self._bp[key]

    def Q(self, det, op, met, key):
        k = (det, op, met, key)
        if k in self._q: return self._q[k]
        v = None
        f3 = E3M / self.ds / det / f"{op}.npz"
        if f3.exists():
            z = np.load(f3)
            if f"{met}|{key}" in z.files: v = z[f"{met}|{key}"]
        if v is None and self.ds == "voc":
            z = np.load(V.WV / "maps" / det / f"{op}.npz")
            if f"{met}|{key}" in z.files: v = z[f"{met}|{key}"]
        if v is None and self.ds == "coco" and met == "map":
            z = np.load(P1 / "maps" / "rest" / det / f"{op}.npz")
            kk = {"rtfull": "full|1500", "rtboot|1500": "boot|1500"}.get(key, key) if det == "rtdetr_l" else key
            if kk in z.files: v = z[kk]
        if v is None and key in ("full", "rtfull"):
            tag = "rt" if det == "rtdetr_l" else "full"; zf = fast(self.ds, det, op, tag)
            if det == "rtdetr_l": smp = np.arange(zf["npig"].shape[0])
            else: smp = np.arange(self.N) if key == "full" else self.rt
            v = 100 * batch_map_wap(zf, smp, np.ones((1, smp.size)), THREADS)[:, 0 if met == "map" else 1]
        if v is None: raise KeyError(f"{self.ds} {det} {op} {met} {key}")
        self._q[k] = np.asarray(v, float); return self._q[k]

    def curve(self, det, c, met, key, j, lv=None):
        lv = list(range(8)) if lv is None else lv; key_bp = key
        r = self.BP(key_bp)[j, [BOPS.index(f"{c}_{l}") for l in lv]]
        q = np.array([self.Q(det, f"{c}_{l}", met, key)[j] for l in lv]); return r, q

    def bd(self, m, det, a, b, met, key, j, lv=None):
        r1, q1 = self.curve(det, a, met, key, j, lv); r2, q2 = self.curve(det, b, met, key, j, lv); return bd(m, r1, q1, r2, q2)

    def bdv(self, m, det, a, b, met, key, lv=None):
        return np.array([self.bd(m, det, a, b, met, key, j, lv) for j in range(len(self.specs(key)))])
