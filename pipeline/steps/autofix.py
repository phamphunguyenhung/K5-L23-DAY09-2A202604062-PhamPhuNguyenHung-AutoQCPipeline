"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


REMOVE_CODES = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}


def _fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def fix(ds, issues, cfg):
    fixes = []
    errs = [i for i in issues if i.severity == ERROR]

    # 1) xoá ảnh (kèm nhãn); issue khác của ảnh đó bị bỏ qua êm
    removed = set()
    for stem in sorted({i.image for i in errs if i.code in REMOVE_CODES}):
        img = ds.image_path(stem)
        if img is None:
            continue
        code = next(i.code for i in errs if i.image == stem and i.code in REMOVE_CODES)
        img.unlink()
        lp = ds.label_path(stem)
        if lp.exists():
            lp.unlink()
        removed.add(stem)
        fixes.append({"image": stem, "line": "", "action": "remove_image", "before": f"{img.name} ({code})", "after": "(đã xoá ảnh + nhãn)"})
    ds._size_cache.clear()

    # 2) sửa dòng nhãn: gom theo file, ghi MỘT lần, dùng số dòng gốc
    per_image = defaultdict(dict)                 # stem -> {line: "drop" | "clip"}
    for i in errs:
        if i.image in removed or i.line is None:
            continue
        if i.code in DROP_CODES:
            per_image[i.image][i.line] = "drop"   # xoá thắng clip
        elif i.code == "BOX_OUT_OF_IMAGE":
            per_image[i.image].setdefault(i.line, "clip")
    for stem in sorted(per_image):
        p = ds.label_path(stem)
        if not p.exists():
            continue
        out = []
        for n, text in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            todo = per_image[stem].get(n)
            if todo is None or not text.strip():
                out.append(text)
                continue
            if todo == "clip":
                try:
                    c, cx, cy, w, h = parse_yolo_line(text)
                    x1, x2 = max(0.0, cx - w / 2), min(1.0, cx + w / 2)
                    y1, y2 = max(0.0, cy - h / 2), min(1.0, cy + h / 2)
                    if x2 - x1 > 1e-6 and y2 - y1 > 1e-6:
                        new = f"{c} {_fmt((x1 + x2) / 2)} {_fmt((y1 + y2) / 2)} {_fmt(x2 - x1)} {_fmt(y2 - y1)}"
                        out.append(new)
                        fixes.append({"image": stem, "line": n, "action": "clip_box", "before": text.strip(), "after": new})
                        continue
                except ValueError:
                    pass
            fixes.append({"image": stem, "line": n, "action": "drop_line", "before": text.strip(), "after": ""})
        p.write_text("".join(l + "\n" for l in out), encoding="utf-8")

    # 3) ảnh chưa có nhãn: không bịa, chuyển người
    for stem in sorted({i.image for i in errs if i.code == "MISSING_LABEL_FILE" and i.image not in removed}):
        fixes.append({"image": stem, "line": "", "action": "needs_label", "before": "(không có file)", "after": "chuyển người gán"})

    # 4) data.yaml
    if any(i.code == "BAD_CLASS_NAME" for i in errs):
        d = {}
        if ds.yaml_path.exists():
            d = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        before = d.get("names")
        d["names"] = dict(enumerate(cfg["classes"]))
        ds.yaml_path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        fixes.append({"image": "data.yaml", "line": "", "action": "rename_classes", "before": str(before), "after": str(cfg["classes"])})
    return fixes
