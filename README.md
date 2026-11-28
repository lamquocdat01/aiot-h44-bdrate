# aiot-h44-bdrate

Code, pre-registrations and derived result files for the manuscript

> **How many images does a BD-rate on detection accuracy need? Test-set sampling error, detector dependence, and the choice of accuracy metric**
> Dat Lam Quoc, Truong Nguyen Xuan, Quang Le Nhut Thanh, Trung Lam Van, Dung Le Trung and Loi Truong Nguyen Thang
> FPT School of Business and Technology (FSB), FPT University, Ho Chi Minh City, Vietnam.
> Version prepared for submission to *Pattern Recognition*: https://github.com/lamquocdat01/aiot-h44-bdrate, tag `submission-pr-v1` (cited in the manuscript). Code, pre-registrations and result files are identical to those of the earlier release tags; only `CITATION.cff` and this README differ.

The study measures the test-set sampling error of the BD-rate computed on object-detection mAP (JPEG, WebP, AVIF at eight quality levels; YOLO26 n/s/l, RT-DETR-L, Faster R-CNN MobileNetV3-Large-FPN), compares it with detector and metric choices, and checks whether a BD-rate gap between detectors changes the quality level chosen for a 1-point mAP-loss target. COCO val2017 was the exploratory set; PASCAL VOC 2007 test was the confirmatory set, analysed under pre-registrations frozen before the corresponding computations.

## Contents
| Path | What |
|---|---|
| `prereg/PREREG_H44_E.md`, `prereg/FREEZE_E.txt` | Pre-registration of hypotheses K1–K4 (Vietnamese original) and its freeze file: time stamp and SHA-256 of the pre-registration, of the analysis code and of three COCO result files, written before any VOC image was encoded |
| `prereg/PREREG_H44_E_C6.md`, `prereg/FREEZE_E_C6.txt` | Pre-registration of claim C6 (operating points) and its freeze file, written before any operating point was computed on VOC |
| `prereg/AMENDMENT_LOG_E.md` | Every post hoc / exploratory analysis and every deviation, with time stamps |
| `code/` | Encoding, detection, evaluation, bootstrap and analysis scripts (Python) |
| `results/*.json` | Result files (summary statistics only; no images, no per-image detections) |
| `code/e4_numbers.py` | Generates every number, table and figure of the manuscript from `results/*.json` |

The registered documents are in Vietnamese; the manuscript (Sections 3.7 and 4.7) and its supplement give the rules in English. This repository was created at submission time from the working directory; its commit history does not reflect the order of the work. The order is documented by the time stamps and SHA-256 values inside `prereg/`.

### Checking the pre-registration hashes
`.gitattributes` keeps every file byte-exact. From the repository root:
```
sha256sum -c prereg/FREEZE_E.txt       # lines for files of the working directory that are not public are reported as missing
sha256sum -c prereg/FREEZE_E_C6.txt
```
`FREEZE_E.txt` lists `results/p1_quality_levels.json`, `results/e2_00_wap_coco.json`, `results/e2_regress_coco.json` and the code files of the confirmatory run; all are included.

## Data (not included, not redistributed)
| Data set | Official source | Use |
|---|---|---|
| COCO val2017 images + `instances_val2017.json` | https://cocodataset.org/ | exploratory phase (4,952 images with at least one non-crowd annotation) |
| PASCAL VOC 2007 test (`VOCtest_06-Nov-2007.tar`) | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ | confirmatory phase (4,952 images) |

Expected layout under a data root given by the environment variable `THS_DATASETS` (default in `code/h44_cfg.py`):
```
<THS_DATASETS>/COCO2017/val2017/*.jpg
<THS_DATASETS>/COCO2017/annotations/instances_val2017.json
<THS_DATASETS>/VOC2007/VOCdevkit/VOC2007/{JPEGImages,Annotations,ImageSets}
```
Encoded images, detections and bootstrap caches are written under `<THS_DATASETS>/H44_pilot` and `<THS_DATASETS>/H44_voc` (tens of GB). Detector weights: `yolo26n.pt`, `yolo26s.pt`, `yolo26l.pt`, `rtdetr-l.pt` from Ultralytics, placed in `<THS_DATASETS>/H44_pilot/weights`; the torchvision Faster R-CNN weights are downloaded by torchvision.

## Environment
Python 3.11.9 on Windows 11, CPU only. `pip install -r requirements.txt` (CPU wheels of torch/torchvision). `code/requirements_h44_full_env.txt` is the complete frozen environment.

## Reproducing
### Tables and figures from the stored results (seconds, no data needed)
```
cd code
python e4_numbers.py ../results <out_dir>     # writes numbers.tex, tables/*.tex, figs/*.pdf, highlights.txt, numbers_trace.json
```

### Re-running the experiments (needs the data; seeds are fixed inside the scripts)
Run times are for a desktop CPU (8 cores).
| Step | Command (from `code/`) | Output | Manuscript |
|---|---|---|---|
| COCO quality levels, encoding | `python s1a_01_select_encode.py`; `python p1_01_encode.py 4` | encoded images, bpp | Table S1 |
| COCO detection + evaluation | `python p1_02_detect.py <nproc> <threads>`; `python p1_03_eval.py build <nproc>`; `python p1_04_draws.py <phase> <nproc>` | per-op evaluations and bootstrap mAP | — |
| COCO exploration | `python e1_01_budget.py`; `python e2_00_wap_coco.py` | `e1_budget.json`, `e2_00_wap_coco.json` | thresholds of K1–K4 |
| VOC confirmation (frozen code) | `python e2_01_voc_encode.py 4`; `python e2_02_voc_detect.py 6`; `python e2_03_voc_eval.py build`; `python e2_04_voc_draws.py 4`; `python e2_05_confirm.py` | `e2_confirm.json` | Table 1 |
| Post hoc | `python e2_06_posthoc.py`; `python e3_02_draws.py`; `python e3_01_f1.py a`, `b`, `c`; `python e3_03_f2f3.py both` | `e2_posthoc.json`, `e3_f1a.json`, `e3_f1bc.json`, `e3_f2.json`, `e3_f3.json` | Tables 2, 3, 5, 6; Figs 1, 2 |
| Extra detectors (exploratory) | `python e3_04_detect.py run yolo26l voc coco --threads 6 --max-hours 13.5`; `python e3_05_f4eval.py build yolo26l voc coco`; `python e3_05_f4eval.py analyse yolo26l`; `python e4_01_frcnn.py run frcnn_mv3 voc --threads 6 --max-hours 9.5`; `python e3_05_f4eval.py build frcnn_mv3 voc`; `python e4_02_g1.py voc` | `e3_f4.json`, `e4_g1.json` | Table 4, Fig. 3 |
| Operating points (C6) | `python e5_c6.py voc`; `python e5_c6.py coco`; `python e5_c6.py extra frcnn_mv3` | `e5_c6.json` | Table 7; Tables S5–S6 |
| Curves for the supplement | `python e4_00_collect.py` | `e4_curves.json` | Tables S1–S2 |

## Licence
Copyright (C) 2026 Dat Lam Quoc, Truong Nguyen Xuan, Quang Le Nhut Thanh, Trung Lam Van, Dung Le Trung and Loi Truong Nguyen Thang.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version (see `LICENSE`). The AGPL was chosen because the code imports Ultralytics.

Third-party components (not distributed here) keep their own licences, as stated in their package metadata: Ultralytics (AGPL-3.0); pillow-heif (BSD-3-Clause) with the bundled libheif (LGPL-3.0) and x265 HEVC encoder (GPL-2.0); pillow-avif-plugin (MIT) with its bundled AVIF libraries; torchvision (BSD-3-Clause); pycocotools (FreeBSD / BSD-2-Clause); PyTorch (BSD-3-Clause and others). Images of COCO and PASCAL VOC are not redistributed; their use is governed by the data set owners' terms and by the terms of the original image sources (Flickr).
