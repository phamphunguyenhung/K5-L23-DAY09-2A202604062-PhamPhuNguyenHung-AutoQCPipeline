from pipeline.checks import aspect_ratio


def got(make_ds, cfg, lines, size=(640, 480)):
    ds = make_ds(images={"a": 1}, labels={"a": lines}, size=size)
    return [(i.code, i.line, i.severity) for i in aspect_ratio.run(ds, cfg)]


def test_normal_boxes_are_silent(make_ds, cfg):
    assert got(make_ds, cfg, ["1 0.5 0.5 0.2 0.3", "0 0.3 0.3 0.4 0.2"]) == []


def test_extreme_ratio_is_warned_with_line_number(make_ds, cfg):
    # dòng 2: 0.8*640=512px x 0.05*480=24px -> tỉ lệ ~21 > 8
    assert got(make_ds, cfg, ["1 0.5 0.5 0.2 0.3", "", "0 0.5 0.5 0.8 0.05"]) == [("BOX_ASPECT", 3, "warn")]


def test_threshold_comes_from_config_and_bad_lines_are_skipped(make_ds, cfg):
    cfg["aspect_ratio"] = {"max_ratio": 30}
    assert got(make_ds, cfg, ["0 0.5 0.5 0.8 0.05", "garbage", "1 0.5 0.5 0 0.3"]) == []
