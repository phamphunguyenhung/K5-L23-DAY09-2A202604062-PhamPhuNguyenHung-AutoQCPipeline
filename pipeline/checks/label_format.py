"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401


def run(ds, cfg):
    c = cfg["label_format"]
    n_cls = len(ds.classes)
    issues = []
    for stem in sorted({p.stem for p in ds.label_files()}):
        lines = ds.read_label_lines(stem) or []      # [(số dòng gốc, nội dung)], đã bỏ dòng trống
        size = ds.image_size(stem)                   # None nếu không có ảnh / ảnh hỏng
        kept = []                                    # (lớp, box) của các dòng hợp lệ ở TRƯỚC, để dò DUP_BOX
        for ln, text in lines:
            def add(sev, code, msg, **extra):
                issues.append(Issue("label_format", sev, code, stem, msg, ln, extra))
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError as e:
                add(ERROR, "BAD_FORMAT", f"dòng không đúng định dạng YOLO: {e}")
                continue
            if not 0 <= cls < n_cls:
                add(ERROR, "BAD_CLASS_ID", f"id lớp {cls} ngoài 0..{n_cls - 1}", id=cls)
                continue
            if w <= 0 or h <= 0:
                add(ERROR, "BOX_ZERO", f"box rỗng (w={w}, h={h})")
                continue
            over = max(0.0, w / 2 - cx, cx + w / 2 - 1, h / 2 - cy, cy + h / 2 - 1)
            if over > c["overflow_tol"]:
                add(ERROR, "BOX_OUT_OF_IMAGE", f"box tràn ảnh {over:.4f}", overflow=round(over, 6))
            box = (cx, cy, w, h)
            if any(k == cls and iou_xywh(b, box) > c["dup_box_iou"] for k, b in kept):
                add(ERROR, "DUP_BOX", "box trùng một dòng trước đó (cùng lớp)")
            kept.append((cls, box))
            if size and min(w * size[0], h * size[1]) < c["min_box_px"]:
                add(WARN, "BOX_TINY", f"cạnh ngắn {min(w * size[0], h * size[1]):.1f}px < {c['min_box_px']}px")
            if w * h > c["max_box_frac"]:
                add(WARN, "BOX_HUGE", f"box chiếm {w * h:.2f} diện tích ảnh")
    return issues
