import collections
from pathlib import Path
from typing import List, Union
import PIL.Image as Image

from tflite_runtime.interpreter import Interpreter

import pycoral.adapters.common as pycoral_common
import pycoral.adapters.classify as classify
from pycoral.adapters.detect import BBox
from pycoral.utils.edgetpu import make_interpreter

Class = collections.namedtuple("Class", ["id", "score"])
Object = collections.namedtuple("Object", ["id", "score", "bbox"])


class SegmentsRunner:
    def __init__(self, segment_paths: List[str], device=None, delegate=None):
        self.segment_paths = segment_paths
        self.interpreters: List[Interpreter] = []
        for segment_path in self.segment_paths:
            self.interpreters.append(
                make_interpreter(
                    segment_path, device=device, delegate=delegate
                )
            )
        self.input_size = pycoral_common.input_size(self.interpreters[0])

        self.initialize()
        self.intermediate = dict()
        self.invoke_cnt = 0

        self.num_segments = len(self.segment_paths)

    def initialize(self):
        for intp in self.interpreters:
            intp.allocate_tensors()
            intp.invoke()

    def allocate_tensors(self):
        for intp in self.interpreters:
            intp.allocate_tensors()

    def set_input(self, data: Union[str, Path]):
        im = Image.open(data).convert("RGB")
        im_resize = im.resize(self.input_size, Image.Resampling.LANCZOS)
        tensor_index = self.interpreters[0].get_input_details()[0]["index"]
        self.interpreters[0].tensor(tensor_index)()[0][:, :] = im_resize

    def set_resized_input(self, image: Image.Image):
        _, scale = pycoral_common.set_resized_input(
            self.interpreters[0],
            image.size,
            lambda size: image.resize(size, Image.LANCZOS),
        )

        return scale

    def invoke_all(self):
        for _ in range(self.num_segments):
            self.invoke()

    def invoke(self) -> bool:
        # 처음 호출 시에는 직접 입력을 넣어준다.
        # 이후에는 `set_intermediate_input`을 통해 이전 결과를 입력으로 넣어준다.
        if self.invoke_cnt != 0:
            self.set_intermediate_input(self.invoke_cnt)

        self.interpreters[self.invoke_cnt].invoke()
        self.invoke_cnt += 1

        # 모든 interpreter를 호출했으면 True를 반환한다.
        if self.invoke_cnt == len(self.interpreters):
            self.invoke_cnt = 0
            return True

        # 아직 모든 interpreter를 호출하지 않았으면,
        # 이전 결과를 intermediate에 저장하고, False를 반환한다.
        else:
            self.update_intermediate(self.invoke_cnt)
            return False

    def set_intermediate_input(self, idx: int):
        input_details = self.interpreters[idx].get_input_details()
        for input_detail in input_details:
            for k, v in self.intermediate.items():
                if input_detail["name"] == k:
                    self.interpreters[idx].set_tensor(input_detail["index"], v)

    def update_intermediate(self, idx):
        output_details = self.interpreters[idx - 1].get_output_details()
        for output_detail in output_details:
            self.intermediate[output_detail["name"]] = self.interpreters[
                idx - 1
            ].get_tensor(output_detail["index"])

    def get_classes(self, top_k=5, score_threshold=0.0) -> List[Class]:
        return classify.get_classes(
            self.interpreters[-1], top_k, score_threshold
        )

    def get_objects(self, score_threshold, image_scale) -> List[Object]:
        def make(i):
            ymin, xmin, ymax, xmax = boxes[i]
            return Object(
                id=int(class_ids[i]),
                score=float(scores[i]),
                bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
                .scale(sx, sy)
                .map(int),
            )

        last_intp = self.interpreters[-1]
        signature_list = last_intp._get_full_signature_list()

        if signature_list:
            if len(signature_list) > 1:
                raise ValueError("Only support model with one signature.")
            signature = signature_list[next(iter(signature_list))]
            count = int(
                last_intp.tensor(signature["outputs"]["output_0"])()[0]
            )
            scores = last_intp.tensor(signature["outputs"]["output_1"])()[0]
            class_ids = last_intp.tensor(signature["outputs"]["output_2"])()[0]
            boxes = last_intp.tensor(signature["outputs"]["output_3"])()[0]
        elif pycoral_common.output_tensor(last_intp, 3).size == 1:
            boxes = pycoral_common.output_tensor(last_intp, 0)[0]
            class_ids = pycoral_common.output_tensor(last_intp, 1)[0]
            scores = pycoral_common.output_tensor(last_intp, 2)[0]
            count = int(pycoral_common.output_tensor(last_intp, 3)[0])
        else:
            scores = pycoral_common.output_tensor(last_intp, 0)[0]
            boxes = pycoral_common.output_tensor(last_intp, 1)[0]
            count = (int)(pycoral_common.output_tensor(last_intp, 2)[0])
            class_ids = pycoral_common.output_tensor(last_intp, 3)[0]

        # NOTE: This becomes a problem when only the last interpreter is used.
        width, height = pycoral_common.input_size(self.interpreters[0])
        image_scale_x, image_scale_y = image_scale
        sx, sy = width / image_scale_x, height / image_scale_y

        return [make(i) for i in range(count) if scores[i] >= score_threshold]
