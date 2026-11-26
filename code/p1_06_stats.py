"""P1 F2.2-F2.5: H1-H4 exactly as in prereg/PREREG_H44_P1.md, lookup table, mAP size bias, secondary results, figures.
Inputs: WORK/p1/maps/{A,rest}/..., WORK/p1/bpp.npz, results/p1_prediction.json.  Output: results/p1_results.json, results/figs/*.pdf
Usage: python p1_06_stats.py"""
import time, platform
import numpy as np
from p1_common import *
from p1_bd import bd_std, bd_iso, computable_std, nonmono
from p1_04_draws import draws, NS_SUB, NS_BOOT, NS_B, NS_RT_SUB, NS_RT_BOOT

TARGETS = [0.15, 0.3, 0.5, 0.8, 1.3, 1.8, 2.4, 3.0]; TOL = 0.15
PAIRS = [("jpeg", "webp"), ("jpeg", "avif"), ("webp", "avif")]
CORE_DETS = ["yolo26n", "yolo26s"]
CNAME = {"jpeg": "JPEG", "webp": "WebP", "avif": "AVIF", "hevc": "HEVC"}
hw = lambda v: float((np.nanpercentile(v, 97.5) - np.nanpercentile(v, 2.5)) / 2)
med = lambda v: float(np.median(v))


def codecs_of(det):
    return CORE + (["hevc"] if det == "yolo26n" else [])


def pairs_of(det):
    return PAIRS + ([(c, "hevc") for c in CORE] if det == "yolo26n" else [])


def load(det, phase):
    ops = ["orig"] + [f"{c}_{l}" for c in codecs_of(det) for l in range(8)]
    Z = {o: np.load(P1 / "maps" / phase / det / f"{o}.npz") for o in ops}
    return ops, {k: np.stack([Z[o][k] for o in ops], 1) for k in Z[ops[0]].files}   # key -> (draws, ops)


def bpp_table(det):
    z = np.load(P1 / "bpp.npz"); s = selection()
    ids = s["rtdetr_subset"] if det == "rtdetr_l" else s["ids"]; pos = {i: k for k, i in enumerate(ids)}
    ops = [f"{c}_{l}" for c in codecs_of(det) for l in range(8)]; B = np.zeros((len(ids), len(ops)))
    oi = {o: j for j, o in enumerate(ops)}
    for i, o, b in zip(z["img"], z["op"], z["bpp"]):
        if o in oi and int(i) in pos: B[pos[int(i)], oi[o]] = b
    return ops, B


def draw_bpp(det, phase, Bm):
    D = draws(phase, det); N = Bm.shape[0]; out = {}
    for k, specs in D.items():
        R = np.zeros((len(specs), Bm.shape[1]))
        for j, (p, kind) in enumerate(specs):
            w = np.bincount(p, minlength=N).astype(float) if kind == "boot" else None
            R[j] = (w @ Bm) / w.sum() if w is not None else Bm[p].mean(0)
        out[k] = R
    return out


def curve_cols(ops, c): return [ops.index(f"{c}_{l}") for l in range(8)]


def analyse_det(det, phase="rest"):
    ops, M = load(det, phase); bops, Bm = bpp_table(det); BP = draw_bpp(det, phase, Bm)
    assert bops == ops[1:]
    res = {"curves": {}, "cells": {}, "full": {}}
    fullkey = [k for k in M if k.startswith("full")][0]
    res["full"]["map"] = dict(zip(ops, M[fullkey][0].tolist()))
    res["full"]["bpp"] = dict(zip(bops, BP[fullkey][0].tolist()))
    keys = [k for k in M if "|" in k]
    for c in codecs_of(det):
        cc = curve_cols(ops, c); cur = {}
        for k in keys:
            q = M[k][:, cc]; cur[k] = {"p_nonmono": float(np.mean([nonmono(x) for x in q])), "n_draws": int(q.shape[0])}
        cur["full_nonmono"] = nonmono(M[fullkey][0][cc]); cur["full_diffs"] = np.diff(M[fullkey][0][cc]).tolist()
        res["curves"][c] = cur
    for a, b in pairs_of(det):
        ca, cb = curve_cols(ops, a), curve_cols(ops, b); cell = {}
        ra, rb = [bops.index(f"{a}_{l}") for l in range(8)], [bops.index(f"{b}_{l}") for l in range(8)]
        for k in keys + [fullkey]:
            Q = M[k]; R = BP[k]; part = k.startswith("Bpart")
            std = np.array([bd_std(R[j, ra], Q[j, ca], R[j, rb], Q[j, cb]) if not part else
                            (0.0 if computable_std(Q[j, ca], Q[j, cb]) else np.nan) for j in range(Q.shape[0])])
            d = {"frac_not_computable_std": float(np.mean(np.isnan(std))), "n_draws": int(Q.shape[0])}
            if not part:
                iso = np.array([bd_iso(R[j, ra], Q[j, ca], R[j, rb], Q[j, cb]) for j in range(Q.shape[0])])
                both = ~np.isnan(std) & ~np.isnan(iso)
                d.update({"frac_not_computable_iso": float(np.mean(np.isnan(iso))), "n_both": int(both.sum()),
                          "absdiff_iso_std": np.abs(iso[both] - std[both]).tolist() if both.sum() else [],
                          "iso_mean": float(np.nanmean(iso)) if (~np.isnan(iso)).any() else None,
                          "iso_hw": hw(iso[~np.isnan(iso)]) if (~np.isnan(iso)).sum() >= 20 else None,
                          "std_hw": hw(std[~np.isnan(std)]) if (~np.isnan(std)).sum() >= 20 else None})
                if k == fullkey: d.update({"bd_std": float(std[0]) if not np.isnan(std[0]) else None, "bd_iso": float(iso[0])})
            cell[k] = d
        res["cells"][f"{a}->{b}"] = cell
    # matched-bpp cells (full-set mean bpp)
    fb = res["full"]["bpp"]; mcells = []
    for T in TARGETS:
        m = {}
        for c in codecs_of(det):
            o = min([f"{c}_{l}" for l in range(8)], key=lambda o: abs(fb[o] - T))
            if abs(fb[o] - T) / T <= TOL: m[c] = o
        for a, b in pairs_of(det):
            if a in m and b in m: mcells.append((T, a, b, m[a], m[b]))
    res["matched"] = []
    for T, a, b, oa, ob in mcells:
        e = {"T": T, "pair": f"{a}->{b}", "ops": [oa, ob]}
        for k in keys:
            if k.startswith(("boot", "sub", "Bboot")):
                dv = M[k][:, ops.index(ob)] - M[k][:, ops.index(oa)]; e[k] = {"hw": hw(dv), "mean": float(dv.mean())}
        e["full_dmap"] = float(M[fullkey][0][ops.index(ob)] - M[fullkey][0][ops.index(oa)])
        res["matched"].append(e)
    # mAP size bias (orig and every op): mean over draws minus full-set value
    res["bias"] = {k: {"orig": float(M[k][:, 0].mean() - M[fullkey][0][0]),
                       "median_all_ops": float(np.median(M[k].mean(0) - M[fullkey][0]))} for k in keys if k.startswith(("sub", "boot"))}
    return res


def main():
    t0 = time.time(); R = {"date": "2026-11-23", "prereg": "prereg/PREREG_H44_P1.md", "seed": SEED_P1}
    A = {det: analyse_det(det) for det in CORE_DETS}; print("core", round(time.time() - t0), flush=True)
    RT = analyse_det("rtdetr_l"); print("rt", round(time.time() - t0), flush=True)
    pred = load_json(RES / "p1_prediction.json")
    core_curves = [(c, d) for d in CORE_DETS for c in CORE]; core_cells = [(f"{a}->{b}", d) for d in CORE_DETS for a, b in PAIRS]
    N = len(selection()["ids"])

    # H1
    nm166 = [A[d]["curves"][c]["sub|166"]["p_nonmono"] for c, d in core_curves]
    nc166 = [A[d]["cells"][k]["sub|166"]["frac_not_computable_std"] for k, d in core_cells]
    H1 = {"nonmono_166_per_curve": dict(zip([f"{c}|{d}" for c, d in core_curves], nm166)), "median_nonmono_166": med(nm166),
          "notcomp_166_per_cell": dict(zip([f"{k}|{d}" for k, d in core_cells], nc166)), "median_notcomp_166": med(nc166)}
    H1["PASS"] = bool(H1["median_nonmono_166"] >= 0.20 and H1["median_notcomp_166"] >= 0.50)

    # H2
    rows = []
    for c, d in core_curves:
        for n in NS_B:
            p = pred["curves"][f"{c}|{d}"][str(n)]; o = A[d]["curves"][c][f"Bpart|{n}"]["p_nonmono"]
            rows.append({"curve": f"{c}|{d}", "n": n, "pred": p, "obs": o, "obs_boot": A[d]["curves"][c][f"Bboot|{n}"]["p_nonmono"]})
    H2 = {"rows": rows, "mae_pp": 100 * float(np.mean([abs(r["pred"] - r["obs"]) for r in rows])),
          "max_abs_pp": 100 * float(np.max([abs(r["pred"] - r["obs"]) for r in rows])),
          "mae_pp_vs_Bboot": 100 * float(np.mean([abs(r["pred"] - r["obs_boot"]) for r in rows])),
          "mean_signed_pp": 100 * float(np.mean([r["pred"] - r["obs"] for r in rows]))}
    H2["mae_pp_by_n"] = {str(n): 100 * float(np.mean([abs(r["pred"] - r["obs"]) for r in rows if r["n"] == n])) for n in NS_B}
    H2["PASS"] = bool(H2["mae_pp"] <= 7.0)
    # first partition only (exactly floor(|B|/n) disjoint subsets) - recomputed from stored per-draw arrays
    ops_d = {}; fp = []
    for c, d in core_curves:
        if d not in ops_d: ops_d[d] = load(d, "rest")
        ops, M = ops_d[d]; cc = curve_cols(ops, c)
        for n in NS_B:
            k = len(selection()["B"]) // n; q = M[f"Bpart|{n}"][:k, cc]
            fp.append(abs(pred["curves"][f"{c}|{d}"][str(n)] - np.mean([nonmono(x) for x in q])))
    H2["mae_pp_first_partition"] = 100 * float(np.mean(fp))
    # secondary: BD non-computability prediction
    bdr = [abs(pred["cells"][f"{k}|{d}"][str(n)] - A[d]["cells"][k][f"Bpart|{n}"]["frac_not_computable_std"]) for k, d in core_cells for n in NS_B]
    H2["secondary_bd_notcomp_mae_pp"] = 100 * float(np.mean(bdr))
    # secondary: hevc curve
    H2["secondary_hevc_yolo26n"] = [{"n": n, "pred": pred["curves"]["hevc|yolo26n"][str(n)], "obs": A["yolo26n"]["curves"]["hevc"][f"Bpart|{n}"]["p_nonmono"]} for n in NS_B]

    # H3
    keys_sub = [f"sub|{n}" for n in NS_SUB]
    H3 = {"iso_notcomp_max_per_n": {}, "median_absdiff_per_n": {}, "n_both_per_n": {}}
    for k in keys_sub + ["full"]:
        H3["iso_notcomp_max_per_n"][k] = max(A[d]["cells"][c][k]["frac_not_computable_iso"] for c, d in core_cells)
        diffs = [x for c, d in core_cells for x in A[d]["cells"][c][k]["absdiff_iso_std"]]
        H3["n_both_per_n"][k] = len(diffs); H3["median_absdiff_per_n"][k] = med(diffs) if len(diffs) >= 20 else None
    H3a = all(v <= 0.01 for v in H3["iso_notcomp_max_per_n"].values())
    H3b = all(v <= 2.0 for v in H3["median_absdiff_per_n"].values() if v is not None)
    H3.update({"a_PASS": bool(H3a), "b_PASS": bool(H3b), "PASS": bool(H3a and H3b)})

    # H4
    NSb = NS_BOOT; H4 = {"dmap": {}, "bd_iso": {}}
    mcore = [(d, e) for d in CORE_DETS for e in A[d]["matched"] if "hevc" not in e["pair"]]
    for n in NSb:
        r = [e[f"boot|{n}"]["hw"] / (e[f"boot|{N}"]["hw"] * np.sqrt(N / n)) for d, e in mcore]
        H4["dmap"][str(n)] = med(r)
        r2 = [A[d]["cells"][k][f"boot|{n}"]["iso_hw"] / (A[d]["cells"][k][f"boot|{N}"]["iso_hw"] * np.sqrt(N / n)) for k, d in core_cells]
        H4["bd_iso"][str(n)] = med(r2)
    H4["n_dmap_cells"] = len(mcore)
    H4["a_PASS"] = bool(all(0.85 <= v <= 1.15 for v in H4["dmap"].values()))
    H4["b_PASS"] = bool(all(0.85 <= v <= 1.15 for v in H4["bd_iso"].values()))
    H4["PASS"] = bool(H4["a_PASS"] and H4["b_PASS"])

    # lookup table (extrapolated from full-set bootstrap)
    hwN_d = np.array([e[f"boot|{N}"]["hw"] for d, e in mcore]); hwN_b = np.array([A[d]["cells"][k][f"boot|{N}"]["iso_hw"] for k, d in core_cells])
    LK = {"dmap": {str(h): {"median": float(N * (np.median(hwN_d) / h) ** 2), "p90": float(N * (np.percentile(hwN_d, 90) / h) ** 2)} for h in (2, 1, 0.5)},
          "bd_iso": {str(h): {"median": float(N * (np.median(hwN_b) / h) ** 2), "p90": float(N * (np.percentile(hwN_b, 90) / h) ** 2)} for h in (10, 5, 2)},
          "hw_at_n": {str(n): {"dmap_median": med([e[f"boot|{n}"]["hw"] for d, e in mcore]), "dmap_p90": float(np.percentile([e[f"boot|{n}"]["hw"] for d, e in mcore], 90)),
                               "bd_iso_median": med([A[d]["cells"][k][f"boot|{n}"]["iso_hw"] for k, d in core_cells])} for n in NSb + [N]}}

    R.update({"H1": H1, "H2": H2, "H3": H3, "H4": H4, "lookup": LK, "per_det": A, "rtdetr_l": RT, "N": N,
              "versions": {"python": platform.python_version(), "numpy": np.__version__}})
    # compact summary tables (supplement)
    R["table_nonmono"] = {f"{c}|{d}": {k: A[d]["curves"][c][k]["p_nonmono"] for k in keys_sub} for d in CORE_DETS for c in codecs_of(d)}
    R["table_nonmono"].update({f"{c}|rtdetr_l": {k: RT["curves"][c][k]["p_nonmono"] for k in RT["curves"][c] if k.startswith("sub")} for c in CORE})
    R["table_notcomp"] = {f"{k}|{d}": {kk: {"std": A[d]["cells"][k][kk]["frac_not_computable_std"], "iso": A[d]["cells"][k][kk]["frac_not_computable_iso"]} for kk in keys_sub}
                          for d in CORE_DETS for k in A[d]["cells"]}
    R["table_notcomp"].update({f"{k}|rtdetr_l": {kk: {"std": v["frac_not_computable_std"], "iso": v["frac_not_computable_iso"]} for kk, v in RT["cells"][k].items() if kk.startswith("sub")} for k in RT["cells"]})
    for d in list(A) + ["rtdetr_l"]:
        X = A[d] if d != "rtdetr_l" else RT
        for k in X["cells"]:
            for kk in X["cells"][k]: X["cells"][k][kk].pop("absdiff_iso_std", None)
    save_json(R, RES / "p1_results.json")
    print("H1", H1["PASS"], H1["median_nonmono_166"], H1["median_notcomp_166"]); print("H2", H2["PASS"], H2["mae_pp"])
    print("H3", H3["PASS"], H3["iso_notcomp_max_per_n"], H3["median_absdiff_per_n"]); print("H4", H4["PASS"], H4["dmap"], H4["bd_iso"])
    figs(R); print("done", round(time.time() - t0))


def figs(R):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "font.family": "serif"})
    F = RES / "figs"; F.mkdir(parents=True, exist_ok=True); A = R["per_det"]
    ns = NS_SUB; mk = {"yolo26n": "o", "yolo26s": "s"}; col = {"jpeg": "C0", "webp": "C1", "avif": "C2", "hevc": "C3"}
    # Fig 1: non-monotone fraction and BD non-computability vs n
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for d in CORE_DETS:
        for c in CORE:
            ax[0].plot(ns, [A[d]["curves"][c][f"sub|{n}"]["p_nonmono"] for n in ns], marker=mk[d], ms=3, color=col[c], lw=0.8, label=f"{CNAME[c]}, {d}")
        for a, b in PAIRS:
            k = f"{a}->{b}"
            ax[1].plot(ns, [A[d]["cells"][k][f"sub|{n}"]["frac_not_computable_std"] for n in ns], marker=mk[d], ms=3, lw=0.8, color="k", alpha=0.6)
            ax[1].plot(ns, [A[d]["cells"][k][f"sub|{n}"]["frac_not_computable_iso"] for n in ns], marker=mk[d], ms=3, lw=0.8, color="C3", alpha=0.6)
    ax[1].plot([], [], color="k", label="standard BD"); ax[1].plot([], [], color="C3", label="isotonic BD")
    for a_ in ax: a_.set_xscale("log"); a_.set_xlabel("test-set size n (images)"); a_.axvline(166, color="0.6", ls=":", lw=0.8)
    ax[0].set_ylabel("fraction non-monotone"); ax[1].set_ylabel("fraction not computable")
    ax[0].legend(fontsize=5.5, ncol=2, frameon=False); ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(F / "fig1_nonmono_bd.pdf"); plt.close(fig)
    # Fig 2: H2 predicted vs observed
    fig, ax = plt.subplots(figsize=(2.4, 2.3))
    for r in R["H2"]["rows"]:
        c, d = r["curve"].split("|"); ax.plot(r["pred"], r["obs"], mk[d], color=col[c], ms=3)
    ax.plot([0, 1], [0, 1], "k--", lw=0.6); ax.set_xlabel("predicted from half A"); ax.set_ylabel("observed on half B")
    fig.tight_layout(); fig.savefig(F / "fig2_pred_obs.pdf"); plt.close(fig)
    # Fig 3: half-width vs n
    N = R["N"]; nb = NS_BOOT + [N]; fig, ax = plt.subplots(figsize=(3.4, 2.3))
    dm = [R["lookup"]["hw_at_n"][str(n)]["dmap_median"] for n in nb]; bi = [R["lookup"]["hw_at_n"][str(n)]["bd_iso_median"] for n in nb]
    ax.loglog(nb, dm, "o-", ms=3, lw=0.8, label=r"$\Delta$mAP at matched bpp (points)")
    ax.loglog(nb, bi, "s-", ms=3, lw=0.8, label="isotonic BD-rate (%)")
    for y in (dm, bi): ax.loglog(nb, [y[-1] * np.sqrt(N / n) for n in nb], "k:", lw=0.6)
    ax.set_xlabel("test-set size n (images)"); ax.set_ylabel("median 95% CI half-width"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(F / "fig3_hw_vs_n.pdf"); plt.close(fig)
    # Fig S1: mAP bias vs n
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for d in CORE_DETS:
        ax.plot(ns, [A[d]["bias"][f"sub|{n}"]["orig"] for n in ns], marker=mk[d], ms=3, lw=0.8, label=f"{d}, subsample")
        ax.plot(NS_BOOT, [A[d]["bias"][f"boot|{n}"]["orig"] for n in NS_BOOT], marker=mk[d], ms=3, lw=0.8, ls="--", label=f"{d}, bootstrap")
    ax.set_xscale("log"); ax.axhline(0, color="0.5", lw=0.5); ax.set_xlabel("n"); ax.set_ylabel("mean mAP(n) - mAP(full), points")
    ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(F / "figS1_map_bias.pdf"); plt.close(fig)


if __name__ == "__main__":
    main()
