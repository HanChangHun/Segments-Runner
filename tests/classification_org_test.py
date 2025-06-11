import time

from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import tflite_runtime.interpreter as tflite


def main():
    org_model_path = "models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
    image_path = "segments_runner/test_data/parrot.jpg"

    labels = read_label_file("segments_runner/test_data/inat_bird_labels.txt")
    delegate_path = "/home/chun/workspaces/tpu/libedgetpus/libedgetpu-separate-cache-inference/out/direct/k8/libedgetpu.so.1.0"
    delegate = tflite.load_delegate(delegate_path)

    interpreter = make_interpreter(org_model_path, delegate=delegate)
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)
    image = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    common.set_input(interpreter, image)

    # Run inference
    print("----INFERENCE TIME----")
    print(
        "Note: The first inference on Edge TPU is slow because it includes",
        "loading the model into Edge TPU memory.",
    )
    for _ in range(10):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        classes = classify.get_classes(interpreter, 1, 0.0)
        print("%.1fms" % (inference_time * 1000))

    print("-------RESULTS--------")
    for c in classes:
        print("%s: %.5f" % (labels.get(c.id, c.id), c.score))


if __name__ == "__main__":
    main()
