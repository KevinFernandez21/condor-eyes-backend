"""Clases del primer prototipo y su correspondencia con COCO."""

from __future__ import annotations

# (nombre, id de categoría COCO, índice en los 80 de `yolov8n.pt`)
CLASSES: tuple[tuple[str, int, int], ...] = (
    ("person", 1, 0),
    ("bicycle", 2, 1),
    ("car", 3, 2),
    ("motorcycle", 4, 3),
    ("bus", 6, 5),
    ("truck", 8, 7),
    ("backpack", 27, 24),
    ("handbag", 31, 26),
    ("suitcase", 33, 28),
)
NAMES: list[str] = [c[0] for c in CLASSES]
COCO_ID_TO_IDX: dict[int, int] = {c[1]: i for i, c in enumerate(CLASSES)}
COCO80_TO_IDX: dict[int, int] = {c[2]: i for i, c in enumerate(CLASSES)}
