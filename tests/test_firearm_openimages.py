"""Tests del conversor Open Images → YOLO del detector de armas."""

from __future__ import annotations

from pathlib import Path

from firearm.openimages import (
    _dedupe,
    _labels,
    read_annotations,
    select,
    write_combined_yaml,
)

HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"


def row(
    img: str, label: str, box=(0.1, 0.3, 0.2, 0.6), group="0", depiction="0"
) -> str:
    x1, x2, y1, y2 = box
    return f"{img},xclick,{label},1,{x1},{x2},{y1},{y2},0,0,{group},{depiction},0\n"


def write_csv(tmp_path: Path, rows: list[str]) -> Path:
    p = tmp_path / "ann.csv"
    p.write_text(HEADER + "".join(rows), encoding="utf-8")
    return p


def test_select_weapons_negatives_and_exclusions(tmp_path: Path):
    p = write_csv(
        tmp_path,
        [
            row("a", "/m/0gxl3"),
            row("a", "/m/01g317"),  # pistola + persona
            row("b", "/m/06c54", depiction="1"),  # rifle dibujado → fuera
            row("c", "/m/06nrc", group="1"),  # escopetas en grupo → fuera
            row("d", "/m/050k8"),
            row("d", "/m/04yx4"),  # teléfono + hombre → negativo
            row("e", "/m/0hnnb"),
            row("e", "/m/04ctx"),  # paraguas + cuchillo → ni arma ni negativo
            row("f", "/m/07k1x"),  # herramienta sola → negativo
        ],
    )
    imgs = read_annotations(p)
    weapon, negative = select(imgs, 10, 10)
    assert weapon == ["a"]
    assert negative == ["d", "f"]
    assert select(imgs, 10, 1)[1] == ["d"]


def test_labels_map_classes_and_dedupe_people(tmp_path: Path):
    p = write_csv(
        tmp_path,
        [
            row("a", "/m/0gxl3", box=(0.4, 0.5, 0.4, 0.5)),
            row("a", "/m/01g317", box=(0.1, 0.3, 0.2, 0.9)),
            row(
                "a", "/m/04yx4", box=(0.1, 0.3, 0.21, 0.9)
            ),  # mismo individuo como "Man"
        ],
    )
    lines = _labels(read_annotations(p)["a"])
    classes = sorted(ln.split()[0] for ln in lines)
    assert classes == ["0", "1"]
    cx, cy, w, h = map(
        float, next(ln for ln in lines if ln.startswith("1")).split()[1:]
    )
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (
        0.45,
        0.45,
        0.1,
        0.1,
    )


def test_dedupe_keeps_distinct_boxes():
    a, b, c = (0, 0, 0.5, 0.5), (0.01, 0, 0.5, 0.5), (0.6, 0.6, 0.9, 0.9)
    assert len(_dedupe([a, b, c])) == 2


def test_combined_yaml(tmp_path: Path):
    y = write_combined_yaml(
        tmp_path, "v2", {"train": ["sim/images/train", "oi/images/train"]}
    ).read_text()
    assert "  - oi/images/train" in y and "1: weapon" in y
