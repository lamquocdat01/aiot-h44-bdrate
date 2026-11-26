"""S1a step 1-2: select 500 COCO val2017 images (>=1 non-crowd label, seed 20261122), 10 fixed groups x 50;
pick 5 quality levels per codec (JPEG/WebP/AVIF via Pillow) on 40 disjoint images to hit mean-bpp targets;
encode all pilot images, record real bpp per image."""
import io, sys, time, platform
import numpy as np
from PIL import Image, features
import pillow_avif  # noqa: F401  (registers AVIF)
from pycocotools.coco import COCO as COCOapi
from h44_cfg import *


def encode(img, codec, q):
    b = io.BytesIO()
    if codec == "jpeg":
        img.save(b, "JPEG", quality=int(q), subsampling=2, optimize=False)
    elif codec == "webp":
        img.save(b, "WEBP", quality=int(q), method=4)
    elif codec == "avif":
        img.save(b, "AVIF", quality=int(q), speed=6)
    return b.getvalue()


def bpp(nbytes, img):
    return nbytes * 8.0 / (img.width * img.height)


def main():
    rng = np.random.default_rng(SEED)
    coco = COCOapi(str(ANN))
    ids = sorted(i for i in coco.getImgIds()
                 if len(coco.getAnnIds(imgIds=i, iscrowd=False)) >= 1)
    perm = rng.permutation(ids)
    pilot = [int(x) for x in perm[:N_IMG]]
    calq = [int(x) for x in perm[N_IMG:N_IMG + N_CAL_Q]]
    groups = {str(g): pilot[g * 50:(g + 1) * 50] for g in range(N_GROUP)}
    save_json({"seed": SEED, "n_eligible": len(ids), "pilot": pilot, "groups": groups,
               "quality_calib_images": calq}, RES / "s1a_selection.json")

    # quality level search (bisection on integer quality, monotone bpp)
    cal_imgs = [Image.open(IMG_DIR / coco.imgs[i]["file_name"]).convert("RGB") for i in calq]
    def mean_bpp(codec, q):
        return float(np.mean([bpp(len(encode(im, codec, q)), im) for im in cal_imgs]))
    levels, achieved = {}, {}
    rng_q = {"jpeg": (1, 95), "webp": (0, 100), "avif": (0, 100)}
    for c in CODECS:
        levels[c], achieved[c] = [], []
        cache = {}
        f = lambda q: cache.setdefault(q, mean_bpp(c, q))
        for t in BPP_TARGETS:
            lo, hi = rng_q[c]
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if f(mid) < t: lo = mid
                else: hi = mid
            q = lo if abs(f(lo) - t) <= abs(f(hi) - t) else hi
            levels[c].append(int(q)); achieved[c].append(f(q))
        print(c, levels[c], [round(a, 3) for a in achieved[c]], flush=True)
    import PIL
    save_json({"targets": BPP_TARGETS, "levels": levels, "calib_mean_bpp": achieved,
               "pillow": PIL.__version__, "avif": features.check("avif"),
               "jpeg_args": "subsampling=4:2:0, optimize=False", "webp_args": "method=4",
               "avif_args": "speed=6", "python": platform.python_version()},
              RES / "s1a_quality_levels.json")

    # encode pilot images
    rows, t0 = [], time.time()
    for k, i in enumerate(pilot):
        im = Image.open(IMG_DIR / coco.imgs[i]["file_name"]).convert("RGB")
        for c in CODECS:
            for li, q in enumerate(levels[c]):
                t = time.perf_counter(); data = encode(im, c, q); te = time.perf_counter() - t
                p = ENC_DIR / f"{c}_{li}" / f"{i:012d}.{c}"
                p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
                rows.append({"img": i, "op": f"{c}_{li}", "codec": c, "q": q, "bytes": len(data),
                             "bpp": bpp(len(data), im), "enc_s": te, "W": im.width, "H": im.height})
        if k % 50 == 0: print("enc", k, round(time.time() - t0), "s", flush=True)
    save_json(rows, RES / "s1a_bpp.json")


if __name__ == "__main__":
    main()
