import time

from PIL import Image
from PIL import ImageDraw

from pycoral.utils.dataset import read_label_file
from segments_runner.segments_runner import SegmentsRunner


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
    segment_paths = [
        "test_data/model/tf2_ssd_mobilenet_v2_coco17_ptq_segment_0_of_3_edgetpu.tflite",
        "test_data/model/tf2_ssd_mobilenet_v2_coco17_ptq_segment_1_of_3_edgetpu.tflite",
        "test_data/model/tf2_ssd_mobilenet_v2_coco17_ptq_segment_2_of_3_edgetpu.tflite",
    ]
    output_path = "test_data/det_output_seg.jpg"

    image_path = "test_data/face.jpg"
    image = Image.open(image_path)

    label_path = "test_data/coco_labels.txt"
    labels = read_label_file(label_path)

    runner = SegmentsRunner(segment_paths, device="pci:0")
    runner.initialize()
    scale = runner.set_resized_input(image)
    print(f"scale: {scale}")

    # Run inference
    print("----INFERENCE TIME----")
    for _ in range(10):
        start = time.perf_counter()
        runner.invoke_all()
        inference_time = time.perf_counter() - start
        objs = runner.get_objects(0.4, scale)
        print("%.2fms" % (inference_time * 1000))

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
