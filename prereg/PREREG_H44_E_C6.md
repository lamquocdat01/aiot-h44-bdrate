# PREREG H44 — khẳng định C6 "BD-rate so với quyết định chọn mức nén" (26-11-2026)
Ghi và đóng băng (`prereg\FREEZE_E_C6.txt`) **trước khi tính bất kỳ q\* nào trên VOC**. Nguồn: `prompts\PROMPT_H44_E_FINISH_20261126.md` mục H0. Mã: `code\e5_c6.py` (đã chạy thử trên COCO, khớp `results\z1_c.json` từng số; chưa chạy trên VOC).

## 1. Thăm dò đã có (COCO, `results\z1_c.json`, đã nhìn)
δ = 1 điểm mAP. JPEG: q\* = 6 ở cả 4 detector; WebP 5/5/5/6; AVIF 4/4/4/7 (yolo26n/s/l / rtdetr_l). Chuyển hồ sơ chéo họ: sụt vượt mục tiêu tối đa 0,58 điểm; 0 vi phạm chắc.

## 2. Giới hạn phải nêu (ghi trước)
Đường mAP theo mức của VOC (từng detector, tập 1.500) **đã được tính và một phần đã xem** ở các bước trước (ví dụ JPEG yolo26n tập đầy đủ 64,22 → 64,15 ở mức cao nhất; bảng BD và mAP ảnh gốc ở `results\e3_f4.json`). Các đại lượng q\*, Δ theo mức so với ảnh gốc trên tập 1.500, và phép chuyển hồ sơ giữa detector **chưa** được tính trên VOC. Vì vậy C6 trên VOC là xác nhận theo nghĩa "luật và ngưỡng cố định trước khi tính", không phải "dữ liệu chưa từng nhìn". Ngưỡng của C6a/C6b đặt sau khi xem COCO.

## 3. Thiết kế (chép prompt, kèm cài đặt)
- Tập: 1.500 ảnh VOC 2007 test của rtdetr_l (`e2_voc_common.rt_subset()`, seed 20261124+210). Detector: yolo26n, yolo26s, yolo26l, rtdetr_l (detection đã lưu; không suy luận mới). Codec JPEG, WebP, AVIF × 8 mức q = 0…7 (q = 7 chất lượng cao nhất), cùng tham số P1.
- mAP = AP@[.5:.95] × 100, COCOeval nhanh đã kiểm (`e3_fastboot`), 20 lớp VOC, `difficult` → crowd.
- Δ_d(c, q) = mAP_d(gốc) − mAP_d(c, q). **q\*_d(c)** = mức thấp nhất mà Δ_d ≤ δ ở chính mức đó và ở mọi mức cao hơn; nếu không có → "không xác định", ô đó tính là **không đạt**.
- CI 95 % percentile của Δ: 1.000 bootstrap ghép cặp đã lưu `rtboot|1500` (seed 20261124+200, luồng `[seed, 1500, 1]`), chung cho mọi detector, ảnh gốc và mọi mức. Không sinh mẫu mới (seed 20261126+600 không dùng cho phép tính nào của C6).
- bpp = bpp trung bình trên tập 1.500.
- **C6a (trong họ YOLO):** 3 cặp detector {n,s}, {n,l}, {s,l} × 3 codec = 9 ô; ô đạt nếu |q\*_A − q\*_B| ≤ 1. **ĐẠT nếu ≥ 8/9 ô.**
- **C6b (chéo họ, yolo26n ↔ rtdetr_l, 3 codec × 2 chiều = 6 ô):** dùng q\*_A cho B. Ô đạt vế 1 nếu Δ_B(q\*_A) − δ ≤ 1,0 (ước lượng điểm). Vi phạm chắc: CI 95 % của Δ_B(q\*_A) nằm hoàn toàn trên δ. **ĐẠT nếu cả 6 ô đạt vế 1 và số vi phạm chắc ≤ 1.**
- δ = 1 là cổng. Báo kèm, không cổng: q\* từng ô; bpp(q\*_A)/bpp(q\*_B); cùng phép tính với δ = 2; hiệu BD (isotonic + PCHIP, L8, tập 1.500) tương ứng đặt cạnh.
- Chấm ĐẠT / KHÔNG ĐẠT nguyên văn. Không đạt → C6 vào bài dưới dạng **kết quả hỗn hợp** (COCO có, VOC không), không bỏ.
- Đầu ra: `results\e5_c6.json` (khoá `voc`, mục `verdict`), `S2\E_C6_H44.md`.
- Chế độ `extra` của mã (detector họ thứ ba ở H1) là **thăm dò**, không vào cổng.
