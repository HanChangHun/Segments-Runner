import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle(
            [(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="red"
        )
        draw.text(
            (bbox.xmin + 10, bbox.ymin + 10),
            "%s\n%.2f" % (labels.get(obj.id, obj.id), obj.score),
            fill="red",
        )


def main():
    image_path = "test_data/face.jpg"
    label_path = "test_data/coco_labels.txt"
    model_path = (
        "test_data/model/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    )
    output_path = "test_data/det_output_org.jpg"

    labels = read_label_file(label_path)
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()

    image = Image.open(image_path)
    _, scale = common.set_resized_input(
        interpreter,
        image.size,
        lambda size: image.resize(size, Image.LANCZOS),
    )
    print(f"scale: {scale}")

    print("----INFERENCE TIME----")
    print(
        "Note: The first inference is slow because it includes",
        "loading the model into Edge TPU memory.",
    )
    for _ in range(5):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, 0.4, scale)
        print("%.2f ms" % (inference_time * 1000))

    print("-------RESULTS--------")
    if not objs:
        print("No objects detected")

    for obj in objs:
        print(labels.get(obj.id, obj.id))
        print("  id:    ", obj.id)
        print("  score: ", obj.score)
        print("  bbox:  ", obj.bbox)

    image = image.convert("RGB")
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save(output_path)


if __name__ == "__main__":
    main()
