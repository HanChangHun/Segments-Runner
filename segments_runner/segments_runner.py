import collections
from pathlib import Path
from typing import List, Union
import PIL.Image as Image

from tflite_runtime.interpreter import Interpreter

from pycoral.utils.edgetpu import make_interpreter
import pycoral.adapters.common as pycoral_common
import pycoral.adapters.classify as classify
from pycoral.adapters import detect

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

    def get_classes(self, top_k=1, score_threshold=0.0) -> List[Class]:
        return classify.get_classes(
            self.interpreters[-1], top_k, score_threshold
        )

    def get_objects(self, scale, threshold=0.4) -> List[Object]:
        return detect.get_objects(self.interpreters[-1], threshold, scale)
