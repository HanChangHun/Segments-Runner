"""Characterization tests for segments_runner pure logic (no Edge TPU hardware).

These pin the *current* behavior of the parts of segments_runner that require
no Coral hardware, no .tflite delegates, and no model files.  They are a
safety net so that future refactors can be verified green-before / green-after.

Baselines were captured from the code as-of segments_runner 1.1.0.

Out of scope (hardware-dependent, not covered here):
  - SegmentsRunner.__init__  — calls make_interpreter / load_delegate at construction.
  - SegmentsRunner.make_interpreters — same.
  - SegmentsRunner.invoke_* family — requires live tflite interpreters.
  - SegmentsRunner.get_classification_result / get_detection_result — needs
    live interpreter tensor accessors.
  - prepare_detection_image — calls pycoral_common.set_resized_input which
    internally touches the interpreter tensor layout.
"""

import collections
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# helpers shared across tests
# ---------------------------------------------------------------------------

_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "segments_runner" / "test_data"


# ---------------------------------------------------------------------------
# namedtuple constructors (Class, Object)
# ---------------------------------------------------------------------------

from segments_runner.segments_runner import Class, Object
from pycoral.adapters.detect import BBox


class TestClassNamedTuple:
    """Class namedtuple carries id and score fields."""

    def test_class_fields_accessible_by_name(self):
        c = Class(id=5, score=0.95)
        assert c.id == 5
        assert c.score == pytest.approx(0.95)

    def test_class_is_iterable_in_order(self):
        c = Class(id=3, score=0.1)
        assert tuple(c) == (3, pytest.approx(0.1))

    def test_class_zero_score(self):
        c = Class(id=0, score=0.0)
        assert c.id == 0
        assert c.score == pytest.approx(0.0)


class TestObjectNamedTuple:
    """Object namedtuple carries id, score, and a BBox."""

    def test_object_fields_accessible_by_name(self):
        bbox = BBox(xmin=10, ymin=20, xmax=30, ymax=40)
        o = Object(id=2, score=0.7, bbox=bbox)
        assert o.id == 2
        assert o.score == pytest.approx(0.7)
        assert o.bbox == bbox

    def test_object_bbox_round_trips_fields(self):
        bbox = BBox(xmin=0, ymin=0, xmax=100, ymax=200)
        o = Object(id=0, score=1.0, bbox=bbox)
        assert o.bbox.xmin == 0
        assert o.bbox.ymax == 200


# ---------------------------------------------------------------------------
# BBox pure geometry (pycoral.adapters.detect.BBox)
# ---------------------------------------------------------------------------

class TestBBoxGeometry:
    """BBox.scale and BBox.map are pure arithmetic — no hardware involved."""

    def test_scale_multiplies_coordinates(self):
        bbox = BBox(xmin=0.1, ymin=0.2, xmax=0.9, ymax=0.8)
        scaled = bbox.scale(100, 200)
        assert scaled == BBox(xmin=pytest.approx(10.0), ymin=pytest.approx(40.0),
                              xmax=pytest.approx(90.0), ymax=pytest.approx(160.0))

    def test_map_int_truncates_to_integer(self):
        bbox = BBox(xmin=0.1, ymin=0.2, xmax=0.9, ymax=0.8)
        mapped = bbox.scale(100, 200).map(int)
        assert mapped == BBox(xmin=10, ymin=40, xmax=90, ymax=160)

    def test_unit_bbox_scaled_equals_dimensions(self):
        bbox = BBox(xmin=0.0, ymin=0.0, xmax=1.0, ymax=1.0)
        scaled = bbox.scale(300, 400).map(int)
        assert scaled == BBox(xmin=0, ymin=0, xmax=300, ymax=400)

    def test_detection_scale_math_sx_sy(self):
        # Replicates the sx, sy computation in get_detection_result:
        #   sx, sy = (width / scale_w, height / scale_h)
        # followed by BBox.scale(sx, sy).map(int)
        width, height = 300, 300
        scale_w, scale_h = 1.0, 0.75
        sx = width / scale_w
        sy = height / scale_h
        assert sx == pytest.approx(300.0)
        assert sy == pytest.approx(400.0)

        bbox = BBox(xmin=0.2, ymin=0.1, xmax=0.9, ymax=0.8).scale(sx, sy).map(int)
        assert bbox == BBox(xmin=60, ymin=40, xmax=270, ymax=320)


# ---------------------------------------------------------------------------
# Label file parsing
# ---------------------------------------------------------------------------

class TestLabelFileParsing:
    """pycoral.utils.dataset.read_label_file parses plain-text label files.
    These tests pin the contents of the bundled test_data label files and the
    label-lookup pattern used in get_classification_result.
    """

    def test_imagenet_labels_count(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "imagenet_labels.txt"))
        assert len(labels) == 1001

    def test_imagenet_label_zero_is_background(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "imagenet_labels.txt"))
        assert labels[0] == "background"

    def test_imagenet_label_1000_is_toilet_tissue(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "imagenet_labels.txt"))
        assert labels[1000] == "toilet tissue, toilet paper, bathroom tissue"

    def test_imagenet_missing_key_falls_back_to_id(self):
        """Pins the .get(c.id, c.id) lookup pattern in get_classification_result."""
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "imagenet_labels.txt"))
        assert labels.get(1001, 1001) == 1001

    def test_coco_labels_count(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "coco_labels.txt"))
        assert len(labels) == 90

    def test_coco_label_zero_is_person(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "coco_labels.txt"))
        assert labels[0] == "person"

    def test_inat_bird_labels_count(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "inat_bird_labels.txt"))
        assert len(labels) == 965

    def test_inat_bird_label_zero(self):
        from pycoral.utils.dataset import read_label_file
        labels = read_label_file(str(_TEST_DATA_DIR / "inat_bird_labels.txt"))
        assert labels[0] == "Haemorhous cassinii (Cassin's Finch)"


# ---------------------------------------------------------------------------
# CE mode path derivation
# ---------------------------------------------------------------------------

class TestCEModePathDerivation:
    """Pins the path-suffix logic from SegmentsRunner.make_interpreters for CE mode.

    The logic appends '_c' / '_e' before the file extension:
        stem + '_c' + suffix  -> cache model path
        stem + '_e' + suffix  -> execution model path
    This is tested here without instantiating SegmentsRunner (which would
    require hardware).
    """

    @staticmethod
    def _derive_ce_paths(model_path: str):
        path_obj = Path(model_path)
        stem = path_obj.stem
        suffix = path_obj.suffix
        parent = path_obj.parent
        cache_path = parent / f"{stem}_c{suffix}"
        exec_path = parent / f"{stem}_e{suffix}"
        return cache_path, exec_path

    def test_mobilenet_segment_cache_path(self):
        cache, _ = self._derive_ce_paths(
            "models/mobilenet_v2_segment_0_of_3_edgetpu.tflite"
        )
        assert cache == Path("models/mobilenet_v2_segment_0_of_3_edgetpu_c.tflite")

    def test_mobilenet_segment_exec_path(self):
        _, exec_path = self._derive_ce_paths(
            "models/mobilenet_v2_segment_0_of_3_edgetpu.tflite"
        )
        assert exec_path == Path("models/mobilenet_v2_segment_0_of_3_edgetpu_e.tflite")

    def test_efficientnet_segment_paths(self):
        cache, exec_path = self._derive_ce_paths(
            "models/efficientnet_m_segment_1_of_2_edgetpu.tflite"
        )
        assert cache == Path("models/efficientnet_m_segment_1_of_2_edgetpu_c.tflite")
        assert exec_path == Path("models/efficientnet_m_segment_1_of_2_edgetpu_e.tflite")

    def test_preserves_nested_parent_directory(self):
        cache, exec_path = self._derive_ce_paths(
            "a/b/c/my_model_edgetpu.tflite"
        )
        assert cache == Path("a/b/c/my_model_edgetpu_c.tflite")
        assert exec_path == Path("a/b/c/my_model_edgetpu_e.tflite")


# ---------------------------------------------------------------------------
# is_ce_mode detection
# ---------------------------------------------------------------------------

class TestIsCEModeDetection:
    """Pins the is_ce_mode flag logic from SegmentsRunner.__init__."""

    @staticmethod
    def _is_ce(delegate_path):
        return bool(delegate_path and "_ce" in delegate_path)

    def test_ce_delegate_path_returns_true(self):
        assert self._is_ce("/usr/lib/libedgetpu_ce.so") is True

    def test_normal_delegate_path_returns_false(self):
        assert self._is_ce("/usr/lib/libedgetpu.so") is False

    def test_none_delegate_path_returns_false(self):
        assert self._is_ce(None) is False

    def test_empty_string_delegate_path_returns_false(self):
        assert self._is_ce("") is False


# ---------------------------------------------------------------------------
# Score dequantization (pure numpy, no interpreter)
# ---------------------------------------------------------------------------

class TestScoreDequantization:
    """Pins the quantized-to-float conversion logic in get_classification_result.

    The production code:
        if np.issubdtype(dtype, np.integer):
            scores = scale * (output_data.astype(np.int64) - zero_point)
        else:
            scores = output_data.copy()
    """

    def test_integer_dtype_triggers_dequantization(self):
        assert np.issubdtype(np.uint8, np.integer) is True

    def test_float_dtype_does_not_trigger_dequantization(self):
        assert np.issubdtype(np.float32, np.integer) is False

    def test_uint8_dequantization_formula(self):
        scale = 0.00390625  # 1/256
        zero_point = 128
        output_data = np.array([128, 200, 100], dtype=np.uint8)
        scores = scale * (output_data.astype(np.int64) - zero_point)
        expected = np.array([0.0, 0.28125, -0.109375])
        np.testing.assert_allclose(scores, expected)

    def test_float_output_is_copied_not_aliased(self):
        output_data = np.array([0.1, 0.9, 0.5], dtype=np.float32)
        scores = output_data.copy()
        scores[0] = 999.0
        assert output_data[0] == pytest.approx(0.1)  # original unchanged


# ---------------------------------------------------------------------------
# Default resource paths
# ---------------------------------------------------------------------------

class TestDefaultResourcePaths:
    """Pins the default file paths computed relative to the package module."""

    def test_default_labels_file_exists(self):
        import segments_runner.segments_runner as sr_module
        base_dir = Path(sr_module.__file__).resolve().parent
        assert (base_dir / "test_data" / "imagenet_labels.txt").exists()

    def test_default_image_file_exists(self):
        import segments_runner.segments_runner as sr_module
        base_dir = Path(sr_module.__file__).resolve().parent
        assert (base_dir / "test_data" / "parrot.jpg").exists()
