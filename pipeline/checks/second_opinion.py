"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401
from ..model_cache import get_preds  # noqa: F401


def run(ds, cfg, preds=None):
    c = cfg["second_opinion"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")
    issues = []
    for p in ds.image_files():
        stem = p.stem
        lines = ds.read_label_lines(stem)
        if stem not in preds or lines is None:
            continue
        size = ds.image_size(stem)
        labels = ds.valid_boxes(stem)

        def add(code, msg, line=None, **extra):
            issues.append(Issue("second_opinion", WARN, code, stem, msg, line, extra))

        for cls, conf, cx, cy, w, h in preds[stem]:
            cls = int(cls)
            if conf < c["conf"]:
                continue
            if size and (w * size[0]) * (h * size[1]) < c["min_area_px"]:
                continue
            r = (cx, cy, w, h)
            cands = [(iou_xywh(r, lb[1:]), lb[0]) for lb in labels]
            cands = [(iou, k) for iou, k in cands if iou >= c["iou_missing"]]
            if not cands:
                add("POSSIBLE_MISSING", f"model thấy lớp {cls} (conf {conf:.2f}) mà nhãn không có box", conf=conf)
                continue
            same = [iou for iou, k in cands if k == cls]
            if same:
                if max(same) < c["iou_loose"]:
                    add("POSSIBLE_LOOSE_BOX", f"box lớp {cls} lỏng/lệch (IoU {max(same):.2f})", conf=conf, iou=round(max(same), 4))
                continue
            other = max(iou for iou, _ in cands)
            if other >= c["iou_match"]:
                add("POSSIBLE_WRONG_CLASS", f"model cho là lớp {cls} (conf {conf:.2f}), nhãn gán lớp khác (IoU {other:.2f})", conf=conf, iou=round(other, 4))

        fp = [pr for pr in preds[stem] if pr[1] >= c["fp_conf"]]
        for ln, text in lines:
            try:
                cls, cx, cy, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            if not 0 <= cls < len(ds.classes):
                continue
            best = max((iou_xywh((cx, cy, w, h), pr[2:6]) for pr in fp), default=0.0)
            if best < c["iou_missing"]:
                add("POSSIBLE_FALSE_POS", f"box lớp {cls} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})", ln, iou=round(best, 4))
    return issues
