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


# 분류/Class, 감지/Object 결과를 저장할 NamedTuple 정의
Class = collections.namedtuple("Class", ["id", "score"])
Object = collections.namedtuple("Object", ["id", "score", "bbox"])


class SegmentsRunner:
    """
    여러 TFLite 모델(세그먼트)을 순차적으로 실행할 수 있는 클래스입니다.
    분류(classification)와 감지(detection) 모델 모두 지원하도록 기능을 확장한 예시입니다.

    Parameters
    ----------
    model_paths : List[str]
        TFLite 모델(세그먼트) 경로 리스트
    labels_path : Optional[str]
        레이블 파일 경로(기본값: None). None이면 test_data 내부의 coco_labels.txt 사용
    input_file : Optional[str]
        입력 이미지 파일 경로(기본값: None). None이면 test_data 내부의 parrot.jpg 사용
    skip_invoke : bool
        True이면 모델을 호출하지 않고(delegate만 로드) 추후 별도 작업 가능
    delegator_path : Optional[str]
        사용자 지정 delegate(.so) 파일 경로. skip_invoke=True일 때 유효
    device : Optional[str]
        make_interpreter(device=...)에 전달할 EdgeTPU device 명칭(USB, PCI 등)
    """

    def __init__(
        self,
        model_paths: List[str],
        labels_path: Optional[str] = None,
        input_file: Optional[str] = None,
        skip_invoke: bool = False,
        delegator_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        # 모델 경로 리스트
        self.model_paths = model_paths
        self.skip_invoke = skip_invoke
        self.delegator_path = delegator_path
        self.device = device

        # 현재 파일 경로: segments_runner.py
        # -> parent: segments_runner 폴더
        # -> parent.parent: 프로젝트 루트(여기에 test_data 폴더 존재)
        self._base_dir = Path(__file__).resolve().parent

        # delegate 설정
        if self.skip_invoke:
            if not self.delegator_path:
                raise ValueError("delegator_path must be provided when skip_invoke is True")
            self.delegate = tflite.load_delegate(self.delegator_path)
        else:
            self.delegate = None

        # 레이블 파일 설정
        if labels_path is None:
            # test_data 내부 coco_labels.txt를 기본값으로 사용
            labels_file = self._base_dir / "test_data" / "coco_labels.txt"
            self.labels = read_label_file(str(labels_file))
        else:
            self.labels = read_label_file(labels_path)

        # 입력 이미지 설정
        if input_file is None:
            # test_data 내부 parrot.jpg를 기본값으로 사용
            image_file = self._base_dir / "test_data" / "parrot.jpg"
            self.image = Image.open(str(image_file))
        else:
            self.image = Image.open(input_file)

        # Interpreter 생성
        self.interpreters: List[tflite.Interpreter] = []
        self._make_interpreters()
        self.num_segments: int = len(self.interpreters)
        self._allocate_tensors_all()

        # 중간 텐서 저장용 dict
        self.intermediate = dict()
        # 현재 실행 중인 세그먼트 인덱스
        self.cur_idx = 0

        # 입력 텐서 정보 (단일 입력 가정)
        self.input_details = self.interpreters[0].get_input_details()
        self.input_tensor_index = self.input_details[0]["index"]
        self._dtype = self.input_details[0]["dtype"]
        self.input_size = pycoral_common.input_size(self.interpreters[0])

        # 출력 텐서 정보 (역시 단일 출력 가정: 분류용)
        self.output_details = self.interpreters[-1].get_output_details()[0]
        self.scale, self.zero_point = self.output_details["quantization"]

        # 분류 전처리용 이미지
        self.proc_image = self._prepare_classification_image(self.image, self._dtype)

        # 감지 전처리용 이미지 & scale
        self._prepare_detection_image(self.image, self._dtype)

    # ----------------------------------------------------------------
    # 내부 준비 로직
    # ----------------------------------------------------------------
    def _make_interpreters(self):
        """
        model_paths에 대해 make_interpreter를 호출하여 self.interpreters를 구성.
        """
        for model_path in self.model_paths:
            interpreter = make_interpreter(
                str(model_path),
                delegate=self.delegate,
                device=self.device,
            )
            self.interpreters.append(interpreter)

    def _allocate_tensors_all(self):
        """모든 Interpreter에 대해 allocate_tensors를 실행한다."""
        for interpreter in self.interpreters:
            interpreter.allocate_tensors()

    def _prepare_classification_image(self, image: Image.Image, dtype):
        """
        분류(Classification)용 이미지를 모델 입력 크기로 리사이즈하고,
        RGB 변환 후 numpy 배열로 만든다.
        """
        try:
            # interpreter의 (배치, 높이, 너비, 채널) 형태
            _, input_h, input_w, _ = self.interpreters[0].get_input_details()[0]["shape"]
            proc_image = image.convert("RGB").resize((input_w, input_h), Image.LANCZOS)
            return np.asarray(proc_image, dtype=dtype)[np.newaxis, :]
        except Exception as e:
            print(f"Error while preparing classification image: {e}")
            shape = self.interpreters[0].get_input_details()[0]["shape"]
            return np.zeros(shape, dtype=dtype)

    def _prepare_detection_image(self, image: Image.Image, dtype):
        """
        감지(detection)용 이미지를 Interpreter의 입력 크기에 맞추어 리사이즈하고,
        내부적으로 scale 정보를 저장한다.
        """
        _, scale = pycoral_common.set_resized_input(
            self.interpreters[0],
            image.size,
            lambda size: image.resize(size, Image.LANCZOS),
        )
        self.det_scale = scale

    # ----------------------------------------------------------------
    # 외부 이미지 설정
    # ----------------------------------------------------------------
    def set_image(self, new_img: Image.Image, detection=False):
        """
        외부에서 이미지를 새로 지정할 때 호출.
        detection=True이면 내부적으로 감지용 이미지(det_image)로 재준비.
        """
        self.image = new_img

        if not self.interpreters:
            return

        _dtype = self.input_details[0]["dtype"]
        if detection:
            self._prepare_detection_image(self.image, _dtype)
        else:
            self.proc_image = self._prepare_classification_image(self.image, _dtype)

    # ----------------------------------------------------------------
    # 모델 실행(여러 세그먼트)
    # ----------------------------------------------------------------
    def invoke_all(self, task=None, profile=False):
        """
        모든 Interpreter(세그먼트)에 대해 순차적으로 invoke를 수행.
        task='detection'이면 감지 모델, 아니면 분류 모델로 가정.
        """
        self.cur_idx = 0
        for _ in range(self.num_segments):
            self.invoke_and_next(task=task, profile=profile)

    def invoke_and_next(self, task=None, profile=False):
        """
        현재 인덱스(cur_idx)에 해당하는 Interpreter를 한 번 실행 후,
        다음 인덱스로 넘어감.

        Returns
        -------
        0 : 아직 마지막 Interpreter가 아님
        1 : 마지막 Interpreter 실행 후 다시 0으로 초기화
        """
        assert self.cur_idx < self.num_segments, "Current index exceeds number of segments."
        self.invoke_idx(self.cur_idx, task=task, profile=profile)

        if self.cur_idx < self.num_segments - 1:
            self.cur_idx += 1
            return 0
        elif self.cur_idx == self.num_segments - 1:
            self.cur_idx = 0
            return 1
        else:
            raise RuntimeError("Index out of range after invoke.")

    def invoke_idx(self, idx, task=None, profile=False):
        """
        주어진 인덱스의 Interpreter를 실행한다.

        Parameters
        ----------
        idx : int
            self.interpreters 내 인덱스
        task : Optional[str]
            "detection"이면 감지용 입력 사용, 아니면 분류용 입력
        profile : bool
            True면 h2d, exec, d2h 각각의 시간을 ms 단위로 측정하여 리스트로 반환
        """
        interpreter = self.interpreters[idx]
        h2d_dur = self._set_input(idx, task=task, profile=profile)
        exec_dur = self._invoke_interpreter(interpreter, profile=profile)
        d2h_dur = self._store_output_tensors(interpreter, profile=profile)

        if profile:
            return [h2d_dur, exec_dur, d2h_dur]

    # ----------------------------------------------------------------
    # 입력/출력 설정 및 저장
    # ----------------------------------------------------------------
    def _set_input(self, idx, task=None, profile=False):
        """idx에 해당하는 Interpreter에 입력을 설정한다."""
        start_time = time.perf_counter() if profile else None

        if idx == 0:
            self._set_initial_input(task=task)
        else:
            self._set_sequential_input(self.interpreters[idx])

        if profile:
            return (time.perf_counter() - start_time) * 1000

    def _set_initial_input(self, task=None):
        """
        첫 번째 Interpreter에 대해 입력을 설정한다.
        task='detection'이면 감지용 이미지를, 아니면 분류용 이미지를 사용한다.
        """
        if task != "detection" and self.proc_image is not None:
            self.interpreters[0].set_tensor(self.input_tensor_index, self.proc_image)
        # detection의 경우 set_resized_input에서 이미 내부 텐서를 맞췄으므로 별도 동작 불필요

    def _set_sequential_input(self, interpreter):
        """
        두 번째 이후 Interpreter에 대해서는 이전 세그먼트의 출력을 입력으로 설정한다.
        """
        input_details = interpreter.get_input_details()
        for input_detail in input_details:
            in_name = input_detail["name"]
            if in_name in self.intermediate:
                interpreter.set_tensor(input_detail["index"], self.intermediate[in_name])

    def _invoke_interpreter(self, interpreter, profile=False):
        """해당 Interpreter를 실제로 invoke한다."""
        start_time = time.perf_counter() if profile else None
        interpreter.invoke()
        if profile:
            return (time.perf_counter() - start_time) * 1000

    def _store_output_tensors(self, interpreter, profile=False):
        """
        실행이 끝난 Interpreter의 출력 텐서를 intermediate 딕셔너리에 저장한다.
        """
        start_time = time.perf_counter() if profile else None

        for output_detail in interpreter.get_output_details():
            self.intermediate[output_detail["name"]] = interpreter.get_tensor(
                output_detail["index"]
            )

        if profile:
            return (time.perf_counter() - start_time) * 1000

    # ----------------------------------------------------------------
    # 결과 추출
    # ----------------------------------------------------------------
    def get_result(self, top_n=1, detection=False, image_scale=(1.0, 1.0), score_threshold=0.4):
        """
        마지막 Interpreter의 결과를 가져와 분류 결과 또는 감지 결과로 해석한다.

        Parameters
        ----------
        top_n : int
            분류 결과에서 상위 N개만 반환
        detection : bool
            True면 감지 결과 반환, False면 분류 결과 반환
        image_scale : tuple(float, float)
            필요 시 사용할 추가 스케일
        score_threshold : float
            감지 모델의 점수 임계값

        Returns
        -------
        dict or list
            - 분류 모델: {레이블: 점수}
            - 감지 모델: Object(namedtuple)의 리스트
        """
        if detection:
            result = self._get_detection_result(score_threshold)
        else:
            result = self._get_classification_result(top_n)

        # 다음 호출 전에 이전 출력을 비워 새 실행에 대비
        self.intermediate = {}
        return result

    def _get_classification_result(self, top_n=1):
        """
        마지막 Interpreter의 출력 데이터를 분류로 해석한다.
        """
        output_data = self.interpreters[-1].tensor(self.output_details["index"])().flatten()

        if np.issubdtype(self.output_details["dtype"], np.integer):
            # int 양자화 모델
            scores = self.scale * (output_data.astype(np.int64) - self.zero_point)
        else:
            # float 모델
            scores = output_data.copy()

        classes = pycoral_classify.get_classes_from_scores(scores, top_n, score_threshold=0.0)
        return {self.labels.get(c.id, c.id): float(c.score) for c in classes}

    def _get_detection_result(self, score_threshold=0.4):
        """
        마지막 Interpreter의 출력 데이터를 감지로 해석한다.
        """
        interpreter = self.interpreters[-1]
        signature_list = interpreter._get_full_signature_list()

        # Signature 검사 (PyCoral SignatureDef 모델, 아니면 기본 Tensor 추출)
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
