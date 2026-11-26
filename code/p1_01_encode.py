"""P1 F1 step 1: encode all eligible COCO val2017 images (>=1 non-crowd label; 4952) with JPEG/WebP/AVIF x 8 levels
(S1a + S1a2 quality settings, unchanged) and HEVC intra (HEIF via pillow-heif/x265) x 8 levels.
HEVC levels tuned on the 50 S1a2 tuning images (bisection on integer quality, mean-bpp targets 0.15..3.0).
Existing pilot files are reused (not re-encoded). Output: WORK/enc/<op>/*, WORK/p1/bpp.npz, results/p1_quality_levels.json.
Usage: python p1_01_encode.py [nproc]"""
import io, sys, time, platform
import numpy as np
from multiprocessing import Pool
from PIL import Image
import PIL, pillow_avif  # noqa
import pillow_heif
pillow_heif.register_heif_opener()
from pycocotools.coco import COCO as COCOapi
from h44_cfg import *
from s1a_01_select_encode import encode as enc3, bpp

P1 = WORK / "p1"
TARGETS8 = [0.15, 0.3, 0.5, 0.8, 1.3, 1.8, 2.4, 3.0]


def eligible_ids():
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCOapi(str(ANN))
    return coco, sorted(i for i in coco.getImgIds() if len(coco.getAnnIds(imgIds=i, iscrowd=False)) >= 1)


def encode(img, codec, q):
    if codec == "hevc":
        b = io.BytesIO(); img.save(b, "HEIF", quality=int(q)); return b.getvalue()
    return enc3(img, codec, q)


def levels_all():
    l1 = load_json(RES / "s1a_quality_levels.json")["levels"]; l2 = load_json(RES / "s1a2_quality_levels.json")["levels"]
    L = {c: l1[c] + l2[c] for c in CODECS}
    f = RES / "p1_quality_levels.json"
    if f.exists(): L["hevc"] = load_json(f)["levels"]["hevc"]
    return L


def tune_hevc(coco):
    tune = load_json(RES / "s1a2_quality_levels.json")["tune_images"]
    imgs = [Image.open(IMG_DIR / coco.imgs[i]["file_name"]).convert("RGB") for i in tune]
    cache = {}
    f = lambda q: cache.setdefault(q, float(np.mean([bpp(len(encode(im, "hevc", q)), im) for im in imgs])))
    lv, got = [], []
    for t in TARGETS8:
        lo, hi = 0, 100
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if f(mid) < t: lo = mid
            else: hi = mid
        q = lo if abs(f(lo) - t) <= abs(f(hi) - t) else hi
        lv.append(int(q)); got.append(f(q)); print("hevc", t, q, round(f(q), 3), flush=True)
    L = levels_all()
    save_json({"targets": TARGETS8, "levels": {**{c: L[c] for c in CODECS}, "hevc": lv}, "hevc_tune_mean_bpp": got,
               "hevc_tune_images": "s1a2 tune_images (50)", "pillow": PIL.__version__, "pillow_heif": pillow_heif.__version__,
               "libheif": pillow_heif.libheif_info(), "hevc_args": "Image.save(format='HEIF', quality=q); defaults (4:2:0, 8-bit)",
               "python": platform.python_version()}, RES / "p1_quality_levels.json")


_G = {}
def work(args):
    i, fn = args
    if "L" not in _G: _G["L"] = levels_all()
    L = _G["L"]; im = Image.open(IMG_DIR / fn).convert("RGB"); out = []
    for c in CODECS + ["hevc"]:
        for li, q in enumerate(L[c]):
            op = f"{c}_{li}"; ext = "heic" if c == "hevc" else c
            p = ENC_DIR / op / f"{i:012d}.{ext}"
            if p.exists(): n = p.stat().st_size; te = float("nan")
            else:
                t = time.perf_counter(); data = encode(im, c, q); te = time.perf_counter() - t
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data); n = len(data)
            out.append((i, op, n, bpp(n, im), te))
    return out


def main():
    coco, ids = eligible_ids(); print("eligible", len(ids), flush=True)
    if not (RES / "p1_quality_levels.json").exists(): tune_hevc(coco)
    L = levels_all(); ops = [f"{c}_{l}" for c in CODECS + ["hevc"] for l in range(8)]
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 4; t0 = time.time(); rows = []
    with Pool(nproc) as pool:
        for k, r in enumerate(pool.imap(work, [(i, coco.imgs[i]["file_name"]) for i in ids], chunksize=4)):
            rows += r
            if k % 200 == 0: print("enc", k, round(time.time() - t0), flush=True)
    img = np.array([r[0] for r in rows]); op = np.array([r[1] for r in rows]); nb = np.array([r[2] for r in rows])
    bp = np.array([r[3] for r in rows]); te = np.array([r[4] for r in rows])
    P1.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(P1 / "bpp.npz", img=img, op=op, bytes=nb, bpp=bp, enc_s=te, ids=np.array(ids))
    print("done", len(rows), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
