"""E step 2 C: encode all VOC 2007 test images with JPEG/WebP/AVIF/HEVC x 8 levels, quality parameters of P1 unchanged
(results/p1_quality_levels.json; encoder functions from p1_01_encode). Resumable (existing files are kept).
Output WV/enc/<op>/<stem>.<ext>, WV/bpp.npz.  Usage: python e2_01_voc_encode.py [nproc<=4]"""
import sys, time
import numpy as np
from multiprocessing import Pool
from PIL import Image
from e2_voc_common import *
from p1_01_encode import encode, levels_all
from s1a_01_select_encode import bpp

_L = {}


def work(i):
    if "L" not in _L: _L["L"] = levels_all()
    L = _L["L"]; im = Image.open(img_path(i, "orig")).convert("RGB"); out = []
    for c in CORE + ["hevc"]:
        for li, q in enumerate(L[c]):
            op = f"{c}_{li}"; p = img_path(i, op)
            if p.exists(): n = p.stat().st_size; te = float("nan")
            else:
                t = time.perf_counter(); data = encode(im, c, q); te = time.perf_counter() - t
                p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_suffix(".tmp"); tmp.write_bytes(data); tmp.replace(p); n = len(data)
            out.append((i, op, n, bpp(n, im), te))
    return out


def main():
    nproc = min(4, int(sys.argv[1]) if len(sys.argv) > 1 else 4); ids = ids_all(); print("images", len(ids), flush=True)
    t0 = time.time(); rows = []
    with Pool(nproc) as pool:
        for k, r in enumerate(pool.imap(work, ids, chunksize=4)):
            rows += r
            if k % 250 == 0: print("enc", k, round(time.time() - t0), flush=True)
    np.savez_compressed(WV / "bpp.npz", img=np.array([r[0] for r in rows]), op=np.array([r[1] for r in rows]),
                        bytes=np.array([r[2] for r in rows]), bpp=np.array([r[3] for r in rows]), enc_s=np.array([r[4] for r in rows]),
                        ids=np.array(ids), levels=str(levels_all()))
    print("done", len(rows), round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
