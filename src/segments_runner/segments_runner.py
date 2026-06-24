import time
import collections
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
from pycoral.adapters import common as pycoral_common
from pycoral.adapters import classify as pycoral_classify
from pycoral.adapters.detect import BBox
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


Class = collections.namedtuple("Class", ["id", "score"])
Object = collections.namedtuple("Object", ["id", "score", "bbox"])


class SegmentsRunner:
    def __init__(
        self,
        model_paths: List[str],
        labels_path: Optional[str] = None,
        input_file: Optional[str] = None,
        delegate_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.model_paths = model_paths
        self.delegate_path = delegate_path
        self.device = device
        self.num_segments: int = len(model_paths)

        # Check if CE (Cache and Execution) mode
        self.is_ce_mode = delegate_path and "_ce" in delegate_path
        self.executing = False  # 모델 실행 중인지 확인 (Segment Switch 시 필요)

        # Initialize interpreters based on mode
        self.interpreters: List[tflite.Interpreter] = []
        if self.is_ce_mode:
            self.cache_interpreters: List[tflite.Interpreter] = []
            self.is_cached: List[bool] = [False] * self.num_segments

        if self.delegate_path:
            _option = {}
            if self.device:
                _option["device"] = self.device
                self.delegate = tflite.load_delegate(self.delegate_path, _option)
            else:
                self.delegate = tflite.load_delegate(self.delegate_path)

        else:
            self.delegate = None

        self._base_dir = Path(__file__).resolve().parent

        if labels_path is None:
            labels_file = self._base_dir / "test_data" / "imagenet_labels.txt"
            self.labels = read_label_file(str(labels_file))
        else:
            self.labels = read_label_file(labels_path)

        if input_file is None:
            image_file = self._base_dir / "test_data" / "parrot.jpg"
            self.image = Image.open(str(image_file))
        else:
            self.image = Image.open(input_file)

        self.make_interpreters()
        self.allocate_tensors_all()

        self.intermediate = dict()
        self.cur_idx = 0

        self.input_details = self.interpreters[0].get_input_details()
        self.input_tensor_index = self.input_details[0]["index"]
        self._dtype = self.input_details[0]["dtype"]
        self.input_size = pycoral_common.input_size(self.interpreters[0])

        self.output_details = self.interpreters[-1].get_output_details()[0]
        self.scale, self.zero_point = self.output_details["quantization"]

        self.proc_image = self.prepare_classification_image(self.image, self._dtype)

        self.prepare_detection_image(self.image)

    def make_interpreters(self):
        if self.is_ce_mode:
            # CE mode: find _c and _e model files
            for model_path in self.model_paths:
                path_obj = Path(model_path)
                stem = path_obj.stem
                suffix = path_obj.suffix
                parent = path_obj.parent

                # Create cache and execution model paths
                cache_path = parent / f"{stem}_c{suffix}"
                exec_path = parent / f"{stem}_e{suffix}"

                # Create cache interpreter
                cache_interp = make_interpreter(str(cache_path), delegate=self.delegate)
                self.cache_interpreters.append(cache_interp)

                # Create execution interpreter (stored in self.interpreters)
                exec_interp = make_interpreter(str(exec_path), delegate=self.delegate)
                self.interpreters.append(exec_interp)
        else:
            # Normal mode
            for model_path in self.model_paths:
                if self.delegate:
                    interpreter = make_interpreter(str(model_path), delegate=self.delegate)
                else:
                    interpreter = make_interpreter(str(model_path), device=self.device)
                self.interpreters.append(interpreter)

    def allocate_tensors_all(self):
        if self.is_ce_mode:
            # Also allocate tensors for cache interpreters
            for cache_interp in self.cache_interpreters:
                cache_interp.allocate_tensors()

        for interpreter in self.interpreters:
            interpreter.allocate_tensors()

    def set_image(self, new_img: Image.Image, detection=False):
        self.image = new_img

        if not self.interpreters:
            return

        _dtype = self.input_details[0]["dtype"]
        if detection:
            self.prepare_detection_image(self.image)
        else:
            self.proc_image = self.prepare_classification_image(self.image, _dtype)

    def prepare_classification_image(self, image: Image.Image, dtype):
        try:
            # interpreter의 (배치, 높이, 너비, 채널) 형태
            _, input_h, input_w, _ = self.interpreters[0].get_input_details()[0]["shape"]
            proc_image = image.convert("RGB").resize((input_w, input_h), Image.LANCZOS)
            return np.asarray(proc_image, dtype=dtype)[np.newaxis, :]
        except Exception as e:
            print(f"Error while preparing classification image: {e}")
            shape = self.interpreters[0].get_input_details()[0]["shape"]
            return np.zeros(shape, dtype=dtype)

    def prepare_detection_image(self, image: Image.Image):
        _, scale = pycoral_common.set_resized_input(
            self.interpreters[0],
            image.size,
            lambda size: image.resize(size, Image.LANCZOS),
        )
        self.det_scale = scale

    def invoke_caching_idx(self, idx, profile=False):
        """Invoke caching interpreter for the given segment index."""
        if not self.is_ce_mode:
            return

        if self.is_cached[idx]:
            return  # Already cached

        cache_intp = self.cache_interpreters[idx]

        start_time = time.perf_counter() if profile else None
        cache_intp.invoke()

        self.is_cached[idx] = True

        if profile:
            return (time.perf_counter() - start_time) * 1000

    def invoke_all(self, task=None, profile=False):
        self.cur_idx = 0
        for _ in range(self.num_segments):
            self.invoke_and_next(task=task, profile=profile)

    def invoke_and_next(self, task=None, profile=False):
        assert self.cur_idx < self.num_segments, "Current index exceeds number of segments."

        if not profile:
            self.invoke_idx(self.cur_idx, task=task)
        else:
            h2d, exec, d2h = self.invoke_idx(self.cur_idx, task=task, profile=profile)
            print(f"[SegmentsRunner] h2d: {h2d}, exec: {exec}, d2h: {d2h}")

        if self.cur_idx < self.num_segments - 1:
            self.cur_idx += 1
            return 0
        elif self.cur_idx == self.num_segments - 1:
            self.cur_idx = 0
            self.executing = False
            return 1
        else:
            raise RuntimeError("Index out of range after invoke.")

    def invoke_idx(self, idx, task=None, profile=False):
        self.executing = True

        # CE mode: invoke caching first if not cached
        if self.is_ce_mode:
            cache_dur = self.invoke_caching_idx(idx, profile=profile)
            if profile:
                print(f"[SegmentsRunner] cache: {cache_dur}")

        interpreter = self.interpreters[idx]
        h2d_dur = self.set_input(idx, task=task, profile=profile)
        exec_dur = self.invoke_interpreter(interpreter, profile=profile)
        d2h_dur = self.store_output_tensors(interpreter, profile=profile)

        if profile:
            return h2d_dur, exec_dur, d2h_dur

    def set_input(self, idx, task=None, profile=False):
        start_time = time.perf_counter() if profile else None

        if idx == 0:
            self.set_initial_input(task=task)
        else:
            self.set_intermediate_input(self.interpreters[idx])

        if profile:
            return (time.perf_counter() - start_time) * 1000

    def set_initial_input(self, task=None):
        if task != "detection" and self.proc_image is not None:
            self.interpreters[0].set_tensor(self.input_tensor_index, self.proc_image)
        # detection의 경우 set_resized_input에서 이미 수행

    def set_intermediate_input(self, interpreter):
        input_details = interpreter.get_input_details()
        for input_detail in input_details:
            in_name = input_detail["name"]
            if in_name in self.intermediate:
                interpreter.set_tensor(input_detail["index"], self.intermediate[in_name])

    def invoke_interpreter(self, interpreter, profile=False):
        start_time = time.perf_counter() if profile else None
        interpreter.invoke()
        if profile:
            return (time.perf_counter() - start_time) * 1000

    def store_output_tensors(self, interpreter, profile=False):
        start_time = time.perf_counter() if profile else None

        for output_detail in interpreter.get_output_details():
            self.intermediate[output_detail["name"]] = interpreter.get_tensor(output_detail["index"])

        if profile:
            return (time.perf_counter() - start_time) * 1000

    def get_result(self, top_n=1, detection=False, score_threshold=0.4):
        if detection:
            result = self.get_detection_result(score_threshold)
        else:
            result = self.get_classification_result(top_n)

        self.intermediate = {}
        return result

    def get_classification_result(self, top_n=1):
        output_data = self.interpreters[-1].tensor(self.output_details["index"])().flatten()

        if np.issubdtype(self.output_details["dtype"], np.integer):
            scores = self.scale * (output_data.astype(np.int64) - self.zero_point)
        else:
            scores = output_data.copy()

        classes = pycoral_classify.get_classes_from_scores(scores, top_n, score_threshold=0.0)
        return {self.labels.get(c.id, c.id): float(c.score) for c in classes}

    def get_detection_result(self, score_threshold=0.4):
        interpreter = self.interpreters[-1]
        signature_list = interpreter._get_full_signature_list()

        if signature_list:
            if len(signature_list) > 1:
                raise ValueError("Only support model with one signature.")
            signature = signature_list[next(iter(signature_list))]
            count = int(interpreter.tensor(signature["outputs"]["output_0"])()[0])
            scores = interpreter.tensor(signature["outputs"]["output_1"])()[0]
            class_ids = interpreter.tensor(signature["outputs"]["output_2"])()[0]
            boxes = interpreter.tensor(signature["outputs"]["output_3"])()[0]
        elif pycoral_common.output_tensor(interpreter, 3).size == 1:
            boxes = pycoral_common.output_tensor(interpreter, 0)[0]
            class_ids = pycoral_common.output_tensor(interpreter, 1)[0]
            scores = pycoral_common.output_tensor(interpreter, 2)[0]
            count = int(pycoral_common.output_tensor(interpreter, 3)[0])
        else:
            scores = pycoral_common.output_tensor(interpreter, 0)[0]
            boxes = pycoral_common.output_tensor(interpreter, 1)[0]
            count = int(pycoral_common.output_tensor(interpreter, 2)[0])
            class_ids = pycoral_common.output_tensor(interpreter, 3)[0]

        width, height = self.input_size
        scale_w, scale_h = self.det_scale
        sx, sy = (width / scale_w, height / scale_h)

        def make_object(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return Object(
                id=int(class_ids[i]),
                score=float(scores[i]),
                bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax).scale(sx, sy).map(int),
            )

        return [make_object(i) for i in range(count) if scores[i] >= score_threshold]
