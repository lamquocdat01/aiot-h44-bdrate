"""E-FINISH G2/H2: every number of the JVCIR manuscript is generated here from results/*.json (no manual typing).
Writes <out>/numbers.tex (macros), <out>/tables/*.tex, <out>/figs/*.pdf and <out>/numbers_trace.json (macro -> json key).
Reads only json files: e2_confirm, e2_00_wap_coco, e2_posthoc, e3_f1a, e3_f1bc, e3_f2, e3_f3, e3_f4, e4_g1 (optional), e5_c6, e4_curves,
p1_quality_levels. Usage: python e4_numbers.py [<results_dir> [<out_dir>]]"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "submission_package" / "JVCIR"
J = lambda n: json.load(open(RES / n, encoding="utf8"))
MAC, TRACE = [], {}
PAIRS = ["jpeg->webp", "jpeg->avif", "webp->avif"]; PN = {"jpeg->webp": "JPEG$\\to$WebP", "jpeg->avif": "JPEG$\\to$AVIF", "webp->avif": "WebP$\\to$AVIF"}
PS = {"jpeg->webp": "J$\\to$W", "jpeg->avif": "J$\\to$A", "webp->avif": "W$\\to$A"}
DN = {"yolo26n": "YOLO26n", "yolo26s": "YOLO26s", "yolo26l": "YOLO26l", "rtdetr_l": "RT-DETR-L", "frcnn_mv3": "FRCNN-MV3"}
DAG = " $\dagger$"
CN = {"jpeg": "JPEG", "webp": "WebP", "avif": "AVIF"}


def f(x, d=1, sign=False):
    if x is None: return "--"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "$-$") if not sign else s.replace("-", "$-$").replace("+", "$+$")


def fi(x): return ("$>2{,}500$" if ">" in x else ("$\leq 50$" if "<" in x else x)) if isinstance(x, str) else f"{int(round(x)):,}".replace(",", "{,}")


def mac(name, val, src):
    assert name.isalpha(), name
    MAC.append(f"\\newcommand{{\\{name}}}{{{val}}}"); TRACE[name] = {"value": val, "source": src}


def ci(c, d=1): return f"[{f(c[0], d)}, {f(c[1], d)}]"


def tab(name, s):
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    s = s.replace(r"\begin{tabular}", r"\begin{adjustbox}{max width=\textwidth}\begin{tabular}").replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
    (OUT / "tables" / f"{name}.tex").write_text(s, encoding="utf8")


def main():
    E2, W0, PH, F1a, F1c, F2, F3, F4, C6, CU, QL = (J(x) for x in ("e2_confirm.json", "e2_00_wap_coco.json", "e2_posthoc.json", "e3_f1a.json", "e3_f1bc.json",
                                                                  "e3_f2.json", "e3_f3.json", "e3_f4.json", "e5_c6.json", "e4_curves.json", "p1_quality_levels.json"))
    G1 = J("e4_g1.json") if (RES / "e4_g1.json").exists() else {"datasets": {}}
    # ---------------- data and design
    mac("Nimg", fi(E2["N"]), "e2_confirm.N"); mac("Nboot", fi(E2["n_boot"]), "e2_confirm.n_boot"); mac("Nrt", fi(1500), "e5_c6.voc.subset")
    for d, k in (("yolo26n", "OrigVocN"), ("yolo26s", "OrigVocS"), ("rtdetr_l", "OrigVocRT")): mac(k, f(E2["map_orig_full"][d]), f"e2_confirm.map_orig_full.{d}")
    mac("OrigCocoN", f(CU["full"]["coco|yolo26n"]["orig"]), "e4_curves.full.coco|yolo26n.orig"); mac("OrigCocoS", f(CU["full"]["coco|yolo26s"]["orig"]), "e4_curves.full.coco|yolo26s.orig")
    bp = [v for c in ("jpeg", "webp", "avif") for v in CU["full"]["voc|yolo26n"][c]["bpp"]] + [v for c in ("jpeg", "webp", "avif") for v in CU["full"]["coco|yolo26n"][c]["bpp"]]
    mac("BppMin", f(min(bp), 2), "e4_curves.full.*.bpp min"); mac("BppMax", f(max(bp), 2), "e4_curves.full.*.bpp max")
    # ---------------- K1-K4 confirmatory (VOC)
    K1 = E2["K1"]["cells"]; hw = [v["hw500"] for v in K1.values()]
    mac("KoneHwMin", f(min(hw)), "e2_confirm.K1.cells.*.hw500"); mac("KoneHwMax", f(max(hw)), "e2_confirm.K1.cells.*.hw500")
    rg = [v["range_fit_full"] for v in K1.values() if v["range_fit_full"] is not None]
    mac("KoneRangeMin", f(min(rg), 2), "e2_confirm.K1 range_fit_full"); mac("KoneRangeMax", f(max(rg), 2), "e2_confirm.K1 range_fit_full")
    mac("KoneNtested", str(len(rg)), "e2_confirm.K1 n cells with >=2 methods"); mac("KonePass", str(E2["K1"]["n_pass"]), "e2_confirm.K1.n_pass")
    adj = PH["K1_part2_isotonic_adjusted"]; ra = [v["range"] for v in adj.values()]
    mac("KoneIsoRangeMin", f(min(ra), 2), "e2_posthoc.K1_part2_isotonic_adjusted.*.range"); mac("KoneIsoRangeMax", f(max(ra), 2), "e2_posthoc.K1_part2_isotonic_adjusted.*.range")
    hc = [F2["grids"]["coco|L8"]["K1"][k]["hw500"] for k in F2["grids"]["coco|L8"]["K1"]]
    mac("KoneCocoHwMin", f(min(hc)), "e3_f2.grids.coco|L8.K1.*.hw500"); mac("KoneCocoHwMax", f(max(hc)), "e3_f2.grids.coco|L8.K1.*.hw500")
    k2 = [v["ratio"] for v in E2["K2"]["cells"].values()]
    mac("KtwoMin", f(min(k2), 2), "e2_confirm.K2.cells.*.ratio"); mac("KtwoMax", f(max(k2), 2), "e2_confirm.K2.cells.*.ratio")
    k2a = [v["ratio"] for g in F2["grids"].values() for v in g["K2"].values()]
    k2v = [v["ratio"] for k, g in F2["grids"].items() if k.startswith("voc") for v in g["K2"].values()]
    k2c = [v["ratio"] for k, g in F2["grids"].items() if k.startswith("coco") for v in g["K2"].values()]
    mac("KtwoAllMin", f(min(k2a), 1), "e3_f2.grids.*.K2.*.ratio"); mac("KtwoAllMax", f(max(k2a), 1), "e3_f2.grids.*.K2.*.ratio")
    mac("KtwoVocMin", f(min(k2v), 2), "e3_f2.grids.voc|*.K2"); mac("KtwoVocMax", f(max(k2v), 2), "e3_f2.grids.voc|*.K2")
    mac("KtwoCocoMin", f(min(k2c), 2), "e3_f2.grids.coco|*.K2"); mac("KtwoCocoMax", f(max(k2c), 2), "e3_f2.grids.coco|*.K2")
    K3 = E2["K3"]["family"]
    for p, nm in (("jpeg->webp", "JW"), ("jpeg->avif", "JA"), ("webp->avif", "WA")):
        mac(f"Kthree{nm}", f(K3[p]["diff"], 2), f"e2_confirm.K3.family.{p}.diff"); mac(f"Kthree{nm}CI", ci(K3[p]["ci95"], 2), f"e2_confirm.K3.family.{p}.ci95")
        mac(f"KthreeIn{nm}", f(E2["K3"]["within_yolo"][p]["absdiff"], 2), f"e2_confirm.K3.within_yolo.{p}.absdiff")
    c3 = F2["grids"]["coco|L8"]["K3a"]; mac("KthreeCocoJW", f(c3["jpeg->webp"]["diff"]), "e3_f2.grids.coco|L8.K3a"); mac("KthreeCocoJA", f(c3["jpeg->avif"]["diff"]), "e3_f2.grids.coco|L8.K3a")
    K4 = E2["K4"]; r4 = list(K4["r_cells_500"].values())
    mac("KfourR", f(K4["median_r_500"], 3), "e2_confirm.K4.median_r_500"); mac("KfourRmin", f(min(r4), 2), "e2_confirm.K4.r_cells_500"); mac("KfourRmax", f(max(r4), 2), "e2_confirm.K4.r_cells_500")
    mac("KfourCocoR", f(float(W0["r_COCO_500"]), 3), "e2_00_wap_coco.r_COCO_500"); mac("KfourNarrow", fi(100 * (1 - K4["median_r_500"])), "1 - e2_confirm.K4.median_r_500")
    rows = []
    for k, v in K1.items():
        p, d = k.split("|"); rows.append(f"K1 & {PN[p]}, {DN[d]} & half-width {f(v['hw500'], 2)}; range {f(v['range_fit_full'], 2) if v['range_fit_full'] is not None else '(< 2 fits)'} & $\\geq 3$ and $>$ range & {'met' if v['pass'] else 'not met'} \\\\")
    for k, v in E2["K2"]["cells"].items():
        d, n = k.split("|"); rows.append(f"K2 & {DN[d]}, $n={n}$ & ratio {f(v['ratio'], 2)} & $\\geq 1.5$ & {'met' if v['pass'] else 'not met'} \\\\")
    for p in ("jpeg->webp", "jpeg->avif"):
        rows.append(f"K3 & {PN[p]}, RT-DETR-L $-$ YOLO26n & {f(K3[p]['diff'], 2)} {ci(K3[p]['ci95'], 2)} & $\\geq 8$, CI $\\not\\ni 0$ & {'met' if K3[p]['pass'] else 'not met'} \\\\")
    for p in PAIRS:
        w = E2["K3"]["within_yolo"][p]; rows.append(f"K3 & {PN[p]}, $|$YOLO26n $-$ YOLO26s$|$ & {f(w['absdiff'], 2)} & $\\leq 6$ & {'met' if w['pass'] else 'not met'} \\\\")
    rows.append(f"K4 & median over 6 cells, $n=500$ & $r={f(K4['median_r_500'], 3)}$ & $\\leq 0.90$ & {'met' if K4['PASS'] else 'not met'} \\\\")
    tab("tab_confirm", "\\begin{table}[t]\\centering\\small\n\\caption{\\textbf{Confirmatory} test of K1--K4 on PASCAL VOC 2007 test (rules frozen before any VOC image was encoded). Half-widths of 95\\,\\% paired-bootstrap intervals of the BD-rate on mAP, in percentage points; isotonic + PCHIP, eight-level grid (L8). Range = max $-$ min of the full-set BD-rate over \\{PCHIP, Akima, isotonic\\}; where fewer than two fits were computable the registered rule counted part two as met (4 of 6 cells; see text).}\\label{tab:confirm}\n"
        "\\begin{tabular}{@{}llllc@{}}\\toprule\nClaim & Cell & Observed & Registered threshold & Result\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C2: half-width vs n (post hoc)
    nc = F2["n_curve"]; av = [nc[f"voc|{p}|yolo26n"]["a"] for p in PAIRS]; ac = [nc[f"coco|{p}|yolo26n"]["a"] for p in PAIRS]
    mac("AvocMin", f(min(av), 2), "e3_f2.n_curve.voc|*|yolo26n.a"); mac("AvocMax", f(max(av), 2), "e3_f2.n_curve.voc|*|yolo26n.a")
    mac("AcocoMin", f(min(ac), 2), "e3_f2.n_curve.coco|*|yolo26n.a"); mac("AcocoMax", f(max(ac), 2), "e3_f2.n_curve.coco|*|yolo26n.a")
    asc = nc["coco|jpeg->webp|yolo26s"]; mac("AcocoSjw", f(asc["a"], 2), "e3_f2.n_curve.coco|jpeg->webp|yolo26s.a"); mac("AcocoSjwCI", ci(asc["a_ci95"], 2), "e3_f2.n_curve.coco|jpeg->webp|yolo26s.a_ci95")
    n5j = [nc[f"{ds}|{p}|yolo26n"]["n_for_hw5"] for ds in ("coco", "voc") for p in PAIRS[:2]]; n5w = [nc[f"{ds}|webp->avif|yolo26n"]["n_for_hw5"] for ds in ("coco", "voc")]
    mac("NfiveJmin", fi(min(n5j)), "e3_f2.n_curve.*|jpeg->*|yolo26n.n_for_hw5"); mac("NfiveJmax", fi(max(n5j)), "e3_f2.n_curve.*|jpeg->*|yolo26n.n_for_hw5")
    mac("NfiveWmin", fi(min(n5w)), "e3_f2.n_curve.*|webp->avif|yolo26n.n_for_hw5"); mac("NfiveWmax", fi(max(n5w)), "e3_f2.n_curve.*|webp->avif|yolo26n.n_for_hw5")
    NS = ["50", "100", "166", "500", "1000", "2500"]; rows = []
    for ds in ("coco", "voc"):
        for p in PAIRS:
            x = nc[f"{ds}|{p}|yolo26n"]; n5 = x["n_for_hw5"]; n2 = x["n_for_hw2"]; sm = x["smallest_n_ci_excl0"]
            rows.append(f"{ds.upper()} & {PS[p]} & " + " & ".join(f(x["hw"][n]) for n in NS) + f" & {f(x['a'], 2)} {ci(x['a_ci95'], 2)} & {fi(n5) if n5 else '$>2{,}500$'} & {fi(n2) if n2 else '$>2{,}500$'} & {sm if sm else 'none'} \\\\")
    tab("tab_n", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{\\textbf{Post hoc.} Half-width (percentage points) of the 95\\,\\% bootstrap interval of the BD-rate versus test-set size $n$ (YOLO26n, mAP, L8, 1{,}000 draws), fitted exponent $a$ of $\\mathrm{hw}=c\\,n^{-a}$ with 95\\,\\% CI, interpolated $n$ for half-width $\\leq 5$ and $\\leq 2$, and smallest $n$ on the grid whose interval excludes 0.}\\label{tab:n}\n"
        "\\setlength{\\tabcolsep}{3pt}\\begin{tabular}{@{}llrrrrrrlrrr@{}}\\toprule\nSet & Pair & 50 & 100 & 166 & 500 & 1000 & 2500 & $a$ [95\\,\\% CI] & hw$\\leq$5 & hw$\\leq$2 & CI$\\not\\ni$0\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C1: bias and coverage (post hoc)
    b = F1a["cells"]; b50 = [abs(b[f"{ds}|{p}|50"]["bias_mean"]) for ds in ("coco", "voc") for p in PAIRS]
    b500 = [abs(b[f"{ds}|{p}|{n}"]["bias_mean"]) for ds in ("coco", "voc") for p in PAIRS for n in ("500", "1000", "2500")]
    mac("BiasMaxFifty", f(max(b50), 1), "e3_f1a.cells.*|50.bias_mean"); mac("BiasMaxFivehundred", f(max(b500), 1), "e3_f1a.cells.*|>=500.bias_mean")
    sd500 = [b[f"{ds}|{p}|500"]["sd"] for ds in ("coco", "voc") for p in PAIRS]; mac("SdFivehundredMin", f(min(sd500), 1), "e3_f1a.cells.*|500.sd"); mac("SdFivehundredMax", f(max(sd500), 1), "e3_f1a.cells.*|500.sd")
    cv = F1c["cells"]; cvv = [v["coverage"] for v in cv.values()]
    mac("CovMin", f(min(cvv), 3), "e3_f1bc.cells.*.coverage"); mac("CovMax", f(max(cvv), 3), "e3_f1bc.cells.*.coverage"); mac("Souter", str(min(v["S_outer"] for v in cv.values())), "e3_f1bc.cells.*.S_outer")
    cj = cv["coco|jpeg->webp|500"]; mac("CovCocoJWlow", str(cj["miss_low"]), "e3_f1bc.cells.coco|jpeg->webp|500.miss_low"); mac("CovCocoJWhigh", str(cj["miss_high"]), "e3_f1bc.cells.coco|jpeg->webp|500.miss_high")
    rows = []
    for ds in ("coco", "voc"):
        for p in PAIRS:
            rows.append(f"{ds.upper()} & {PS[p]} & " + " & ".join(f(b[f'{ds}|{p}|{n}']['bias_mean'], 2, True) + f" ({f(b[f'{ds}|{p}|{n}']['mc_se_mean'], 2)})" for n in ("50", "166", "500", "2500")) + f" & {f(b[f'{ds}|{p}|500']['sd'], 1)} \\\\")
    rows2 = []
    for k in ("coco|jpeg->webp|166", "coco|webp->avif|166", "coco|jpeg->webp|500", "coco|webp->avif|500", "voc|jpeg->webp|166", "voc|webp->avif|166", "voc|jpeg->webp|500", "voc|webp->avif|500"):
        ds, p, n = k.split("|"); v = cv[k]
        rows2.append(f"{ds.upper()} & {PS[p]} & {n} & {f(v['coverage'], 3)} {ci(v['wilson95'], 3)} & {v['miss_low']} / {v['miss_high']} & {f(v['mean_halfwidth'], 1)} \\\\")
    tab("tab_bias", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{\\textbf{Post hoc.} (a) Bias of the BD-rate at test-set size $n$: mean of BD$_n-$BD$_{\\mathrm{full}}$ over 1{,}000 subsamples without replacement (Monte-Carlo SE in parentheses), with the sampling SD at $n=500$. (b) Coverage of the 95\\,\\% percentile bootstrap interval for BD$_{\\mathrm{full}}$ (" + str(min(v['S_outer'] for v in cv.values())) + " outer samples $\\times$ 300 inner bootstrap draws; Wilson 95\\,\\% interval; misses below/above). YOLO26n, mAP, isotonic L8.}\\label{tab:bias}\n"
        "\\begin{tabular}{@{}llrrrrr@{}}\\toprule\n\\multicolumn{7}{@{}l}{(a) Bias (percentage points)}\\\\\nSet & Pair & $n=50$ & $n=166$ & $n=500$ & $n=2500$ & SD$_{500}$\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\midrule\n\\multicolumn{7}{@{}l}{(b) Coverage}\\\\\nSet & Pair & $n$ & \\multicolumn{2}{l}{coverage [Wilson]} & misses & mean hw\\\\\\midrule\n" +
        "\n".join(r.replace(" & " + r.split(" & ")[3], " & \\multicolumn{2}{l}{" + r.split(" & ")[3] + "}", 1) for r in rows2) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C3: detectors (exploratory)
    rows = []; dsets = F4["datasets"]
    for ds in ("coco", "voc"):
        r = dsets[ds]; g = G1["datasets"].get(ds)
        mods = ["yolo26n", "yolo26s", "yolo26l", "rtdetr_l"] + (["frcnn_mv3"] if g else [])
        for m in mods:
            om = (g["orig_map"][m] if (g and m == "frcnn_mv3") else r["orig_map"][m]); bd = (g["bd"][m] if (g and m == "frcnn_mv3") else r["bd"][m])
            rows.append(f"{ds.upper()} & {DN[m]} & {f(om)} & " + " & ".join(f(bd[p], 1) for p in PAIRS) + " \\\\")
        rows.append("\\addlinespace")
    rows2 = []
    for ds in ("coco", "voc"):
        P = dsets[ds]["paired"]; g = G1["datasets"].get(ds)
        for key, lab in (("yolo26l-yolo26n", "YOLO26l $-$ YOLO26n"), ("rtdetr_l-yolo26n", "RT-DETR-L $-$ YOLO26n"), ("rtdetr_l-yolo26l", "RT-DETR-L $-$ YOLO26l")):
            rows2.append(f"{ds.upper()} & {lab} & " + " & ".join(f"{f(P[f'{key}|{p}']['diff'], 1, True)} {ci(P[f'{key}|{p}']['ci95'], 1)}" for p in PAIRS) + " \\\\")
        if g:
            for key, lab in (("frcnn_mv3-yolo26n", "FRCNN-MV3 $-$ YOLO26n"), ("frcnn_mv3-rtdetr_l", "FRCNN-MV3 $-$ RT-DETR-L")):
                rows2.append(f"{ds.upper()} & {lab} & " + " & ".join(f"{f(g['paired'][f'{key}|{p}']['diff'], 1, True)} {ci(g['paired'][f'{key}|{p}']['ci95'], 1)}" for p in PAIRS) + " \\\\")
    tab("tab_det", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{\\textbf{Exploratory} (both datasets seen; only the RT-DETR-L $-$ YOLO26n rows on VOC for J$\\to$W and J$\\to$A are confirmatory, K3). Detector dependence on the 1{,}500-image subset of each dataset: mAP on the original images and BD-rate (\\%, isotonic L8), then paired differences with 95\\,\\% CI from the same 1{,}000 bootstrap draws." + ("" if G1["datasets"].get("coco") else " FRCNN-MV3 was run on VOC only (compute budget).") + "}\\label{tab:det}\n"
        "\\begin{tabular}{@{}llrrrr@{}}\\toprule\nSet & Detector & mAP (orig.) & J$\\to$W & J$\\to$A & W$\\to$A\\\\\\midrule\n" + "\n".join(rows[:-1]) +
        "\n\\midrule\nSet & Paired difference & \\multicolumn{1}{l}{J$\\to$W} & \\multicolumn{1}{l}{J$\\to$A} & \\multicolumn{2}{l}{W$\\to$A}\\\\\\midrule\n" +
        "\n".join(r.rsplit(" & ", 1)[0] + " & \\multicolumn{2}{l}{" + r.rsplit(" & ", 1)[1].replace(" \\\\", "") + "} \\\\" for r in rows2) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    for ds in ("coco", "voc"):
        P = dsets[ds]["paired"]; D = ds.capitalize()
        ln = [abs(P[f"yolo26l-yolo26n|{p}"]["diff"]) for p in PAIRS[:2]]; rl = [abs(P[f"rtdetr_l-yolo26l|{p}"]["diff"]) for p in PAIRS[:2]]
        mac(f"LmN{D}Max", f(max(ln), 1), f"e3_f4.datasets.{ds}.paired.yolo26l-yolo26n|J*"); mac(f"RtmL{D}Min", f(min(rl), 1), f"e3_f4.datasets.{ds}.paired.rtdetr_l-yolo26l|J*")
        mac(f"OrigL{D}", f(dsets[ds]["orig_map"]["yolo26l"]), f"e3_f4.datasets.{ds}.orig_map.yolo26l"); mac(f"OrigRt{D}", f(dsets[ds]["orig_map"]["rtdetr_l"]), f"e3_f4.datasets.{ds}.orig_map.rtdetr_l")
    mac("LmNCocoWA", f(dsets["coco"]["paired"]["yolo26l-yolo26n|webp->avif"]["diff"], 1), "e3_f4.datasets.coco.paired.yolo26l-yolo26n|webp->avif.diff")
    mac("LmNCocoWACI", ci(dsets["coco"]["paired"]["yolo26l-yolo26n|webp->avif"]["ci95"], 1), "e3_f4.datasets.coco.paired.yolo26l-yolo26n|webp->avif.ci95")
    for ds in ("voc", "coco"):
        g = G1["datasets"].get(ds); D = ds.capitalize()
        if g:
            mac(f"Fr{D}Orig", f(g["orig_map"]["frcnn_mv3"]), f"e4_g1.datasets.{ds}.orig_map.frcnn_mv3")
            for p, nm in (("jpeg->webp", "JW"), ("jpeg->avif", "JA"), ("webp->avif", "WA")):
                x = g["paired"][f"frcnn_mv3-yolo26n|{p}"]; mac(f"Fr{D}{nm}", f(x["diff"], 1, True), f"e4_g1.datasets.{ds}.paired.frcnn_mv3-yolo26n|{p}.diff"); mac(f"Fr{D}{nm}CI", ci(x["ci95"], 1), f"e4_g1.datasets.{ds}.paired.frcnn_mv3-yolo26n|{p}.ci95")
                y = g["paired"][f"frcnn_mv3-rtdetr_l|{p}"]; mac(f"FrRt{D}{nm}", f(y["diff"], 1, True), f"e4_g1.datasets.{ds}.paired.frcnn_mv3-rtdetr_l|{p}.diff")
            mac(f"Fr{D}Verdict", g["rule"]["verdict"], f"e4_g1.datasets.{ds}.rule.verdict")
        else:
            mac(f"Fr{D}Verdict", "not run", "e4_g1 missing")
            for nm in ("Orig", "JW", "JWCI", "JA", "JACI", "WA", "WACI"): mac(f"Fr{D}{nm}", "not run", "e4_g1 has no such dataset")
            for nm in ("JW", "JA", "WA"): mac(f"FrRt{D}{nm}", "not run", "e4_g1 has no such dataset")
    gv = G1["datasets"].get("voc", {}).get("rule", {}).get("verdict_key")
    S3 = {"sep": "On VOC, the third family (a two-stage Faster R-CNN with a MobileNetV3 backbone) stayed within 6 points of YOLO26n on both JPEG pairs, so among the five models it is RT-DETR-L that separates from the CNN detectors.",
          "dep": "On VOC, the third family (a two-stage Faster R-CNN with a MobileNetV3 backbone, FRCNN-MV3) also differed from YOLO26n by at least 8 points on both JPEG pairs, with intervals excluding zero (exploratory reading rule fixed before the run), so the BD-rate depends on the detector and not only on RT-DETR-L; here it is the YOLO26 models that separate from the other two detectors.",
          "unc": "On VOC, the third family (a two-stage Faster R-CNN with a MobileNetV3 backbone) met neither pre-stated reading rule, so these data do not tell whether RT-DETR-L is a special case or one instance of a general detector effect.",
          None: "[G1 PENDING]"}
    mac("CthreeSentence", S3[gv], "e4_g1.datasets.voc.rule.verdict_key")
    # ---------------- C4: wAP (post hoc)
    c = F3["cells"]; dif = [v["diff"] for v in c.values()]
    mac("WapDiffMax", f(max(dif), 1), "e3_f3.cells.*.diff"); mac("WapFlips", str(sum(v["sign_flip"] for v in c.values())), "e3_f3.cells.*.sign_flip")
    mac("WapRankChanges", str(sum(v["changed"] for v in F3["ranking"].values())), "e3_f3.ranking.*.changed")
    jd = [abs(v["diff"]) for k, v in c.items() if k.split("|")[1].startswith("jpeg->")]; mac("WapJpegMax", f(max(jd), 1), "e3_f3.cells.*|jpeg->*.diff")
    wb = [v["diff"] for k, v in c.items() if k.split("|")[1].startswith("webp->")]; mac("WapWebpMin", f(min(wb), 1), "e3_f3.cells.*|webp->*.diff"); mac("WapWebpMax", f(max(wb), 1), "e3_f3.cells.*|webp->*.diff")
    sh = F3["class_share_top5"]; mac("PersonCoco", fi(100 * sh["coco"]["share_top1"]), "e3_f3.class_share_top5.coco.share_top1"); mac("PersonVoc", fi(100 * sh["voc"]["share_top1"]), "e3_f3.class_share_top5.voc.share_top1")
    mac("TopFiveCoco", fi(100 * sh["coco"]["share_top5"]), "e3_f3.class_share_top5.coco.share_top5"); mac("TopFiveVoc", fi(100 * sh["voc"]["share_top5"]), "e3_f3.class_share_top5.voc.share_top5")
    cw = c["coco|webp->avif|yolo26n"]; mac("WapCocoWAmap", f(cw["bd_map"], 1), "e3_f3.cells.coco|webp->avif|yolo26n.bd_map"); mac("WapCocoWAwap", f(cw["bd_wap"], 1, True), "e3_f3.cells.coco|webp->avif|yolo26n.bd_wap")
    rows = []
    for ds in ("coco", "voc"):
        for d in ("yolo26n", "yolo26s"):
            for p in PAIRS:
                v = c[f"{ds}|{p}|{d}"]; rows.append(f"{ds.upper()} & {DN[d]} & {PS[p]} & {f(v['bd_map'], 1)} & {f(v['bd_wap'], 1)} & {f(v['diff'], 1, True)} {ci(v['diff_ci95'], 1)}{DAG if v['sign_flip'] else ''} \\\\")
    tab("tab_wap", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{\\textbf{Post hoc.} BD-rate (\\%) on mAP and on instance-weighted AP (wAP) over the full test sets (4{,}952 images each), and their difference with 95\\,\\% CI from 1{,}000 full-size bootstrap draws. $\\dagger$ = sign change. HEVC (HEIC) cells for YOLO26n are in the supplement.}\\label{tab:wap}\n"
        "\\begin{tabular}{@{}lllrrl@{}}\\toprule\nSet & Detector & Pair & BD (mAP) & BD (wAP) & wAP $-$ mAP [95\\,\\% CI]\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C5: grids (post hoc)
    rows = []
    for ds in ("coco", "voc"):
        for gname in ("L4", "L5", "L6", "L7", "L8"):
            g = F2["grids"][f"{ds}|{gname}"]; s = g["summary"]; r2 = [v["ratio"] for v in g["K2"].values()]
            rows.append(f"{ds.upper()} & {gname} & {s['K1_pass']}/6 ({s['K1_tested']}) & {f(min(r2), 2)}--{f(max(r2), 2)} & {f(g['K3a']['jpeg->webp']['diff'], 1)}; {f(g['K3a']['jpeg->avif']['diff'], 1)} & {f(s['K4_median_r'], 3)} \\\\")
    nchg = sum(len(v) for k in ("changed_vs_L8|coco", "changed_vs_L8|voc") for v in F2[k].values()); mac("GridChanges", str(nchg), "e3_f2.changed_vs_L8|*")
    tab("tab_grid", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{\\textbf{Post hoc.} K1--K4 quantities recomputed on nested grids L4--L8 (the $k$ lowest quality levels). K1: cells meeting the rule (cells where part two was actually tested). K2: half-width ratio WebP$\\to$AVIF / JPEG$\\to$WebP. K3: RT-DETR-L $-$ YOLO26n (J$\\to$W; J$\\to$A, 1{,}500 images). K4: median wAP/mAP half-width ratio at $n=500$.}\\label{tab:grid}\n"
        "\\begin{tabular}{@{}llllll@{}}\\toprule\nSet & Grid & K1 met (tested) & K2 ratio & K3 & K4 $r$\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C6 (VOC confirmatory, COCO exploratory)
    V6, C6c = C6["voc"], C6["coco"]
    mac("CsixaVoc", f"{V6['C6a']['1']['n_ok']}/{V6['C6a']['1']['n']}", "e5_c6.voc.C6a.1"); mac("CsixaCoco", f"{C6c['C6a']['1']['n_ok']}/{C6c['C6a']['1']['n']}", "e5_c6.coco.C6a.1")
    mac("CsixbVocOver", f(V6["C6b"]["1"]["max_overshoot"], 2), "e5_c6.voc.C6b.1.max_overshoot"); mac("CsixbCocoOver", f(C6c["C6b"]["1"]["max_overshoot"], 2), "e5_c6.coco.C6b.1.max_overshoot")
    mac("CsixbVocSure", str(V6["C6b"]["1"]["n_sure_violation"]), "e5_c6.voc.C6b.1.n_sure_violation"); mac("CsixbCocoSure", str(C6c["C6b"]["1"]["n_sure_violation"]), "e5_c6.coco.C6b.1.n_sure_violation")
    mac("CsixVerdictA", "met" if V6["verdict"]["C6a"] == "DAT" else "not met", "e5_c6.voc.verdict.C6a"); mac("CsixVerdictB", "met" if V6["verdict"]["C6b"] == "DAT" else "not met", "e5_c6.voc.verdict.C6b")
    mac("CsixbVocOverTwo", f(V6["C6b"]["2"]["max_overshoot"], 2), "e5_c6.voc.C6b.2.max_overshoot")
    dq = lambda R: max(abs(c["qA"] - c["qB"]) for c in R["C6b"]["1"]["cells"]); mac("CsixMaxQgapVoc", str(dq(V6)), "e5_c6.voc.C6b.1.cells |qA-qB|"); mac("CsixMaxQgapCoco", str(dq(C6c)), "e5_c6.coco.C6b.1.cells |qA-qB|")
    gaps = [V6["bd_gap_rt_minus_n"][p] for p in PAIRS[:2]] + [C6c["bd_gap_rt_minus_n"][p] for p in PAIRS[:2]]
    mac("CsixBdGapMin", f(min(gaps), 0), "e5_c6.*.bd_gap_rt_minus_n J*"); mac("CsixBdGapMax", f(max(gaps), 0), "e5_c6.*.bd_gap_rt_minus_n J*")
    br = [c["bpp_ratio_qA_over_qB"] for R in (V6, C6c) for c in R["C6b"]["1"]["cells"]]; mac("CsixBppRatioMax", f(max(br), 2), "e5_c6.*.C6b.1.cells.bpp_ratio"); mac("CsixBppRatioMin", f(min(br), 2), "e5_c6.*.C6b.1.cells.bpp_ratio")
    qsets = sorted(set(v for R in (V6, C6c) for v in R["qstar"]["1"].values() if v is not None)); mac("CsixQmin", str(qsets[0]), "e5_c6.*.qstar.1"); mac("CsixQmax", str(qsets[-1]), "e5_c6.*.qstar.1")
    EX = C6.get("exploratory_extra", {})
    rows = []
    for ds, R in (("voc", V6), ("coco", C6c)):
        for cdc in ("jpeg", "webp", "avif"):
            qs = [R["qstar"]["1"][f"{d}|{cdc}"] for d in ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l")]
            fx = EX.get(ds, {}).get("qstar", {}).get("1", {}).get(f"frcnn_mv3|{cdc}", "--") if EX else "--"
            rows.append(f"{ds.upper()} & {CN[cdc]} & " + " & ".join(str(q) if q is not None else "none" for q in qs) + f" & {fx if fx is not None else 'none'} \\\\")
    rows2 = []
    for ds, R in (("voc", V6), ("coco", C6c)):
        for cc in R["C6b"]["1"]["cells"]:
            rows2.append(f"{ds.upper()} & {DN[cc['A']]} $\\to$ {DN[cc['B']]} & {CN[cc['codec']]} & {cc['qA']} $\\to$ {cc['qB']} & {f(cc['dB_at_qA'], 2)} {ci(cc['ci_dB_at_qA'], 2)} & {f(cc['dB_at_qB'], 2)} & {f(cc['bpp_ratio_qA_over_qB'], 2)} \\\\")
    tab("tab_c6", "\\begin{table}[t]\\centering\\footnotesize\n\\caption{From BD-rate to operating points. Top: lowest quality level $q^*$ (0--7) whose mAP loss versus the original is $\\leq\\delta=1$ point at that and every higher level (1{,}500-image subset). Bottom: transfer of $q^*$ between detector families; $\\Delta_B(q^*_A)$ = loss of detector $B$ at the level chosen with detector $A$ (95\\,\\% CI, paired bootstrap), $\\Delta_B(q^*_B)$ = its loss at its own level, and the bpp ratio. VOC rows: \\textbf{confirmatory} (C6, separate pre-registration); COCO rows and FRCNN-MV3: \\textbf{exploratory}.}\\label{tab:c6}\n"
        "\\begin{tabular}{@{}lllllll@{}}\\toprule\nSet & Codec & YOLO26n & YOLO26s & YOLO26l & RT-DETR-L & FRCNN-MV3\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\midrule\nSet & $A\\to B$ & Codec & $q^*_A\\to q^*_B$ & $\\Delta_B(q^*_A)$ [95\\,\\% CI] & $\\Delta_B(q^*_B)$ & bpp ratio\\\\\\midrule\n" + "\n".join(rows2) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- C6 exploratory: third family (FRCNN-MV3, VOC only)
    XV = EX.get("voc", {}) if EX else {}
    if XV.get("transfer"):
        T = XV["transfer"]; into = [v for k, v in T.items() if k.endswith("->frcnn_mv3|" + k.split("|")[1] + "|1")]
        outof = [v for k, v in T.items() if k.startswith("frcnn_mv3->") and k.endswith("|1")]
        fq = [XV["qstar"]["1"][f"frcnn_mv3|{c}"] for c in ("jpeg", "webp", "avif")]
        oq = [XV["qstar"]["1"][f"{d}|{c}"] for d in ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l") for c in ("jpeg", "webp", "avif")]
        mac("FrQmin", str(min(fq)), "e5_c6.exploratory_extra.voc.qstar.1.frcnn_mv3|*"); mac("FrQmax", str(max(fq)), "e5_c6.exploratory_extra.voc.qstar.1.frcnn_mv3|*")
        mac("FrOthQmin", str(min(oq)), "e5_c6.exploratory_extra.voc.qstar.1.<4 registered>"); mac("FrOthQmax", str(max(oq)), "e5_c6.exploratory_extra.voc.qstar.1.<4 registered>")
        gap = [abs(XV["qstar"]["1"][f"frcnn_mv3|{c}"] - XV["qstar"]["1"][f"{d}|{c}"]) for d in ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l") for c in ("jpeg", "webp", "avif")]
        mac("FrQgapMin", str(min(gap)), "e5_c6.exploratory_extra.voc.qstar.1 |q_fr-q_d|"); mac("FrQgapMax", str(max(gap)), "e5_c6.exploratory_extra.voc.qstar.1 |q_fr-q_d|")
        mac("FrIntoOver", f(max(v["overshoot"] for v in into), 2), "e5_c6.exploratory_extra.voc.transfer.*->frcnn_mv3|*|1.overshoot max")
        mac("FrIntoSure", str(sum(bool(v["sure_violation"]) for v in into)), "e5_c6.exploratory_extra.voc.transfer.*->frcnn_mv3|*|1.sure_violation count")
        mac("FrIntoN", str(len(into)), "e5_c6.exploratory_extra.voc.transfer.*->frcnn_mv3|*|1 count")
        mac("FrOutSure", str(sum(bool(v["sure_violation"]) for v in outof)), "e5_c6.exploratory_extra.voc.transfer.frcnn_mv3->*|*|1.sure_violation count")
        ob = [v["bpp_ratio_qA_over_qB"] for v in outof]
        mac("FrOutBppMin", f(min(ob), 2), "e5_c6.exploratory_extra.voc.transfer.frcnn_mv3->*|*|1.bpp_ratio"); mac("FrOutBppMax", f(max(ob), 2), "e5_c6.exploratory_extra.voc.transfer.frcnn_mv3->*|*|1.bpp_ratio")
        rows3 = []
        for k, v in T.items():
            a, b = k.split("|")[0].split("->"); cdc, dl = k.split("|")[1], k.split("|")[2]
            rows3.append(f"{dl} & {DN[a]} $\\to$ {DN[b]} & {CN[cdc]} & {v['qA']} $\\to$ {v['qB']} & {f(v['dB_at_qA'], 2)} {ci(v['ci_dB_at_qA'], 2)} & {'yes' if v['sure_violation'] else 'no'} & {f(v['bpp_ratio_qA_over_qB'], 2)} \\\\")
        tab("supp_c6fr", r"\begin{table}[h]\centering\footnotesize\caption{\textbf{Exploratory} (not pre-registered). Transfer of $q^*$ between FRCNN-MV3 and the four registered detectors on VOC (1{,}500-image subset). Sure = 95\,\% CI of $\Delta_B(q^*_A)$ entirely above $\delta$.}\label{tab:sc6fr}" + "\n"
            r"\begin{tabular}{@{}llllllr@{}}\toprule" + "\n" + r"$\delta$ & $A\to B$ & Codec & $q^*_A\to q^*_B$ & $\Delta_B(q^*_A)$ [95\,\% CI] & Sure & bpp ratio\\\midrule" + "\n" + "\n".join(rows3) + "\n" + r"\bottomrule\end{tabular}\end{table}" + "\n")
    else:
        for nm in ("FrQmin", "FrQmax", "FrOthQmin", "FrOthQmax", "FrQgapMin", "FrQgapMax", "FrIntoOver", "FrIntoSure", "FrIntoN", "FrOutSure", "FrOutBppMin", "FrOutBppMax"): mac(nm, "[G1 PENDING]", "e5_c6 extra missing")
    # ---------------- supplement tables
    rows = []
    for ds in ("coco", "voc"):
        for cdc in ("jpeg", "webp", "avif"):
            rows.append(f"{ds.upper()} & {CN[cdc]} & " + " & ".join(f(x, 3) for x in CU["full"][f"{ds}|yolo26n"][cdc]["bpp"]) + " \\\\")
    lv = QL["levels"]
    tab("supp_bpp", "\\begin{table}[h]\\centering\\footnotesize\\caption{Mean bpp per quality level over the full test sets (4{,}952 images). Encoder quality parameters (same for both datasets): " + "; ".join(f"{CN[c]} {lv[c]}" for c in ("jpeg", "webp", "avif")) + ".}\\label{tab:sbpp}\n\\begin{tabular}{@{}ll" + "r" * 8 + "@{}}\\toprule\nSet & Codec & " + " & ".join(f"$q={i}$" for i in range(8)) + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    rows = []
    for ds in ("coco", "voc"):
        for d in ("yolo26n", "yolo26s"):
            x = CU["full"][f"{ds}|{d}"]
            for cdc in ("jpeg", "webp", "avif"): rows.append(f"{ds.upper()} & {DN[d]} ({f(x['orig'], 1)}) & {CN[cdc]} & " + " & ".join(f(m, 1) for m in x[cdc]["map"]) + " \\\\")
    tab("supp_map", "\\begin{table}[h]\\centering\\footnotesize\\caption{mAP@[.5:.95] ($\\times 100$) per quality level on the full test sets; original-image mAP in parentheses.}\\label{tab:smap}\n\\begin{tabular}{@{}lll" + "r" * 8 + "@{}}\\toprule\nSet & Detector & Codec & " + " & ".join(f"$q={i}$" for i in range(8)) + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    rows = []
    for ds, R in (("voc", V6), ("coco", C6c)):
        for d in ("yolo26n", "yolo26s", "yolo26l", "rtdetr_l"):
            for cdc in ("jpeg", "webp", "avif"): rows.append(f"{ds.upper()} & {DN[d]} & {CN[cdc]} & " + " & ".join(f(v, 2) for v in R["delta_full"][f"{d}|{cdc}"]) + " \\\\")
        for d, x in EX.get(ds, {}).get("delta_full", {}).items():
            rows.append(f"{ds.upper()} & {DN[d.split('|')[0]]} & {CN[d.split('|')[1]]} & " + " & ".join(f(v, 2) for v in x) + " \\\\")
    tab("supp_delta", "\\begin{table}[h]\\centering\\footnotesize\\caption{mAP loss $\\Delta_d(c,q)$ versus the original image on the 1{,}500-image subset (points), used to determine $q^*$ (VOC: confirmatory for the four registered detectors).}\\label{tab:sdelta}\n\\begin{tabular}{@{}lll" + "r" * 8 + "@{}}\\toprule\nSet & Detector & Codec & " + " & ".join(f"$q={i}$" for i in range(8)) + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    rows = []
    for ds in ("coco", "voc"):
        for gname in ("L4", "L5", "L6", "L7", "L8"):
            g = F2["grids"][f"{ds}|{gname}"]
            k1 = " & ".join(f(g["K1"][f"{p}|{d}"]["hw500"], 2) + ("" if g["K1"][f"{p}|{d}"]["tested"] else "*") for d in ("yolo26n", "yolo26s") for p in PAIRS)
            k2 = " & ".join(f(g["K2"][f"{d}|{n}"]["ratio"], 2) for d in ("yolo26n", "yolo26s") for n in ("166", "500"))
            rows.append(f"{ds.upper()} & {gname} & {k1} & {k2} & {f(g['K3a']['jpeg->webp']['diff'], 2)} {ci(g['K3a']['jpeg->webp']['ci95'], 1)} & {f(g['K3a']['jpeg->avif']['diff'], 2)} {ci(g['K3a']['jpeg->avif']['ci95'], 1)} \\\\")
    tab("supp_grid", "\\begin{table}[h]\\centering\\footnotesize\\caption{\\textbf{Post hoc.} K1--K3 quantities per cell and grid. K1: half-width at $n=500$ for J$\\to$W, J$\\to$A, W$\\to$A (YOLO26n, then YOLO26s); * = part two of K1 not testable (fewer than two fits). K2: half-width ratio at $n=166$, $500$ (YOLO26n, YOLO26s). K3: RT-DETR-L $-$ YOLO26n with 95\\,\\% CI.}\\label{tab:sgrid}\n\\begin{tabular}{@{}ll" + "r" * 6 + "r" * 4 + "ll@{}}\\toprule\nSet & Grid & \\multicolumn{6}{l}{K1 half-width (n: JW JA WA; s: JW JA WA)} & \\multicolumn{4}{l}{K2 ratio (n166 n500 s166 s500)} & K3 J$\\to$W & K3 J$\\to$A\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    rows = []
    for k, v in sorted(c.items()):
        ds, p, d = k.split("|"); a, bb = p.split("->"); rows.append(f"{ds.upper()} & {DN[d]} & {a.upper()}$\\to${bb.upper()} & {f(v['bd_map'], 2)} & {f(v['bd_wap'], 2)} & {f(v['diff'], 2, True)} {ci(v['diff_ci95'], 2)} \\\\")
    tab("supp_wap", "\\begin{table}[h]\\centering\\footnotesize\\caption{All mAP/wAP BD-rate cells including HEVC (HEIC, YOLO26n).}\\label{tab:swap}\n\\begin{tabular}{@{}llllll@{}}\\toprule\nSet & Detector & Pair & BD mAP & BD wAP & difference [95\\,\\% CI]\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    # ---------------- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42})
    (OUT / "figs").mkdir(parents=True, exist_ok=True)
    st = {"jpeg->webp": ("k", "o", "-"), "jpeg->avif": ("0.45", "s", "--"), "webp->avif": ("k", "^", ":")}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for i, ds in enumerate(("coco", "voc")):
        for p in PAIRS:
            x = nc[f"{ds}|{p}|yolo26n"]; n = [int(k) for k in NS]; y = [x["hw"][k] for k in NS]; col, mk, ls = st[p]
            ax[i].plot(n, y, color=col, marker=mk, ls=ls, ms=4, lw=1, label=f"{p.replace('->', ' to ').replace('jpeg', 'JPEG').replace('webp', 'WebP').replace('avif', 'AVIF')} (a = {x['a']:.2f})")
        ax[i].plot([50, 2500], [20, 20 * (50 / 2500) ** 0.5], color="0.6", lw=0.8, ls="-.", label="slope -1/2")
        ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_xlabel("test-set size n (images)"); ax[i].set_title(f"{'COCO val2017' if ds == 'coco' else 'VOC 2007 test'} (YOLO26n)")
        ax[i].legend(frameon=False, fontsize=6.5); ax[i].grid(True, which="both", lw=0.3, color="0.85")
    ax[0].set_ylabel("half-width of 95% CI (pp)"); fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_hw_n.pdf"); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for i, ds in enumerate(("coco", "voc")):
        P = dsets[ds]["paired"]; g = G1["datasets"].get(ds)
        items = [("YOLO26l", P, "yolo26l-yolo26n"), ("RT-DETR-L", P, "rtdetr_l-yolo26n")] + ([("FRCNN-MV3", g["paired"], "frcnn_mv3-yolo26n")] if g else [])
        for j, p in enumerate(PAIRS):
            for k, (lab, src, key) in enumerate(items):
                v = src[f"{key}|{p}"]; xx = j + (k - (len(items) - 1) / 2) * 0.22
                ax[i].errorbar(xx, v["diff"], yerr=[[v["diff"] - v["ci95"][0]], [v["ci95"][1] - v["diff"]]], fmt=["o", "s", "D"][k], color=["0.55", "k", "0.3"][k], mfc=["0.55", "k", "w"][k], ms=4, capsize=2, lw=0.9, label=lab if j == 0 else None)
        ax[i].axhline(0, color="0.5", lw=0.6); ax[i].set_xticks(range(3)); ax[i].set_xticklabels(["JPEG to WebP", "JPEG to AVIF", "WebP to AVIF"])
        ax[i].set_title(f"{'COCO' if ds == 'coco' else 'VOC'} (1,500 images)"); ax[i].grid(True, axis="y", lw=0.3, color="0.85"); ax[i].legend(frameon=False, fontsize=6.5)
    ax[0].set_ylabel("BD-rate minus YOLO26n BD-rate (pp)"); fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_det.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.2, 2.3)); keys = list(cv)
    order = sorted(keys, key=lambda k: (k.split("|")[0], k.split("|")[2], k.split("|")[1]))
    for j, k in enumerate(order):
        v = cv[k]; ax.errorbar(j, v["coverage"], yerr=[[v["coverage"] - v["wilson95"][0]], [v["wilson95"][1] - v["coverage"]]], fmt="o" if "jpeg" in k else "s", color="k", mfc="k" if "jpeg" in k else "w", ms=4, capsize=2, lw=0.9)
    ax.axhline(0.95, color="0.5", ls="--", lw=0.8); ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{k.split('|')[0].upper()}\n{'J-W' if 'jpeg' in k else 'W-A'}\nn={k.split('|')[2]}" for k in order], fontsize=6)
    ax.set_ylabel("coverage of 95% CI"); ax.set_ylim(0.88, 1.0); ax.grid(True, axis="y", lw=0.3, color="0.85"); fig.tight_layout(); fig.savefig(OUT / "figs" / "fig_cov.pdf"); plt.close(fig)
    H = [f"At 500 test images, a BD-rate on mAP has a 95% CI half-width of {min(hw):.1f} to {max(hw):.1f} points",
         f"Bootstrap intervals were nearly unbiased and covered in {min(cvv):.2f} to {max(cvv):.2f} of resamples",
         (lambda v: f"Two non-YOLO detectors moved the BD-rate by {min(v):.0f} to {max(v):.0f} points; a larger YOLO26 did not")([K3[p]['diff'] for p in PAIRS[:2]] + [G1["datasets"]["voc"]["paired"][f"frcnn_mv3-yolo26n|{p}"]["diff"] for p in PAIRS[:2]]),
         f"A 1-point-loss quality level moved at most {dq(V6)} level between YOLO26 and RT-DETR-L",
         "Weighted AP narrows the interval partly by measuring a different quantity"]
    for h in H: assert len(h) <= 85, (len(h), h)
    (OUT / "highlights.txt").write_text("\n".join(H) + "\n", encoding="utf8"); TRACE["highlights"] = {"value": H, "source": "e2_confirm K1/K3, e3_f1bc, e5_c6"}
    (OUT / "numbers.tex").write_text("% GENERATED by code/e4_numbers.py from results/*.json -- do not edit\n" + "\n".join(MAC) + "\n", encoding="utf8")
    json.dump(TRACE, open(OUT / "numbers_trace.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
    print(len(MAC), "macros;", len(list((OUT / "tables").glob("*.tex"))), "tables; G1 datasets:", list(G1["datasets"]))


if __name__ == "__main__":
    main()
