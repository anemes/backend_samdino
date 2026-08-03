"""Tests for LabelStore fixes: bulk race condition (#1), class validation (#22c)."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hitl.data.label_store import LabelStore, SegClass


def _make_store(tmp: Path) -> LabelStore:
    store = LabelStore(tmp / "labels.gpkg")
    store.set_classes([
        SegClass(class_id=2, name="building", color="#FF0000"),
        SegClass(class_id=3, name="road", color="#00FF00"),
    ])
    store.add_region(mapping(box(0, 0, 10, 10)), crs="EPSG:4326")
    return store


def test_bulk_add_concurrent():
    """#1: Two concurrent bulk adds must produce unique, contiguous IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(Path(tmp))

        def bulk_add():
            geoms = [box(1 + i * 0.1, 1, 1.09 + i * 0.1, 1.09) for i in range(5)]
            store.add_annotations_bulk(
                geometries=geoms,
                class_ids=[2] * 5,
                region_id=1,
            )

        threads = [threading.Thread(target=bulk_add) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        anns = store.get_annotations()
        assert len(anns) == 10, f"Expected 10 annotations, got {len(anns)}"
        ids = sorted(anns["ann_id"].tolist())
        assert len(set(ids)) == 10, f"Duplicate ann_ids: {ids}"
        assert ids == list(range(1, 11)), f"Non-contiguous IDs: {ids}"


def test_bulk_add_single_thread():
    """Regression: single-threaded bulk add still works correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(Path(tmp))

        geoms = [box(1 + i * 0.1, 1, 1.09 + i * 0.1, 1.09) for i in range(3)]
        count = store.add_annotations_bulk(
            geometries=geoms,
            class_ids=[2, 3, 2],
            region_id=1,
        )
        assert count == 3
        anns = store.get_annotations()
        assert len(anns) == 3
        assert sorted(anns["ann_id"].tolist()) == [1, 2, 3]


def test_mixed_single_and_bulk():
    """Regression: single add followed by bulk add produces correct IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _make_store(Path(tmp))

        # Single add
        idx = store.add_annotation(
            geometry_geojson=mapping(box(1, 1, 2, 2)),
            class_id=2,
            region_id=1,
        )
        assert idx == 1

        # Bulk add
        geoms = [box(3, 3, 4, 4), box(5, 5, 6, 6)]
        count = store.add_annotations_bulk(
            geometries=geoms,
            class_ids=[2, 3],
            region_id=1,
        )
        assert count == 2

        anns = store.get_annotations()
        assert len(anns) == 3
        assert sorted(anns["ann_id"].tolist()) == [1, 2, 3]


def test_class_set_and_get():
    """Regression: class CRUD round-trips correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        store = LabelStore(Path(tmp) / "labels.gpkg")

        # Initially empty
        assert store.get_classes() == []

        # Set classes
        classes = [
            SegClass(class_id=2, name="building", color="#FF0000"),
            SegClass(class_id=3, name="road", color="#00FF00"),
        ]
        store.set_classes(classes)
        result = store.get_classes()
        assert len(result) == 2
        assert result[0].name == "building"
        assert result[1].class_id == 3

        # Overwrite with fewer classes
        store.set_classes([SegClass(class_id=2, name="building", color="#FF0000")])
        assert len(store.get_classes()) == 1


def test_class_id_validation():
    """#22c: class_id < 2 must be rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        store = LabelStore(Path(tmp) / "labels.gpkg")
        try:
            store.set_classes([SegClass(class_id=1, name="bad", color="#000")])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


def test_get_num_classes():
    """Regression: get_num_classes returns max(class_id) + 1."""
    with tempfile.TemporaryDirectory() as tmp:
        store = LabelStore(Path(tmp) / "labels.gpkg")
        assert store.get_num_classes() == 2  # ignore(0) + background(1)

        store.set_classes([
            SegClass(class_id=2, name="a", color="#F00"),
            SegClass(class_id=5, name="b", color="#0F0"),
        ])
        assert store.get_num_classes() == 6  # 0..5
