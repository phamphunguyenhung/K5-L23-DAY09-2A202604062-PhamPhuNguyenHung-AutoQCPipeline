"""CHECK thưởng — Box có tỉ lệ khung hình cực đoan (BOX_ASPECT, WARN).

Box quá dài/mảnh (cạnh dài / cạnh ngắn tính bằng px > `max_ratio`) thường là nhãn kéo lệch hoặc đo nhầm cạnh.
Chỉ cảnh báo (người xem ảnh mới biết đúng hay sai). Ngưỡng ở `cfg["aspect_ratio"]["max_ratio"]`, mặc định 8.0
(không có trong configs/pipeline.yaml vì file đó bị ghim; mặc định nằm ở đây).
Chưa đăng ký vào QC_CHECKS vì `pipeline/checks/__init__.py` bị ghim; gọi bằng `aspect_ratio.run(ds, cfg)`.
"""
from ..common import WARN, Issue, parse_yolo_line

DEFAULT_MAX_RATIO = 8.0


def run(ds, cfg):
    max_ratio = cfg.get("aspect_ratio", {}).get("max_ratio", DEFAULT_MAX_RATIO)
    issues = []
    for p in ds.image_files():
        size = ds.image_size(p.stem)
        lines = ds.read_label_lines(p.stem)
        if not size or lines is None:
            continue
        for ln, text in lines:
            try:
                _, _, _, w, h = parse_yolo_line(text)
            except ValueError:
                continue
            wpx, hpx = w * size[0], h * size[1]
            if wpx <= 0 or hpx <= 0:
                continue
            ratio = max(wpx, hpx) / min(wpx, hpx)
            if ratio > max_ratio:
                issues.append(Issue("aspect_ratio", WARN, "BOX_ASPECT", p.stem,
                                    f"tỉ lệ cạnh dài/ngắn {ratio:.1f} > {max_ratio}", ln, {"ratio": round(ratio, 2)}))
    return issues
