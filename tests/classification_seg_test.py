import time
from PIL import Image
from pycoral.utils.dataset import read_label_file
from segments_runner import SegmentsRunner


def main():
    image_path = "segments_runner/test_data/parrot.jpg"
    segment_paths = [
        "models/mobilenet_v2_1.0_224_inat_bird_quant_segment_0_of_3_edgetpu.tflite",
        "models/mobilenet_v2_1.0_224_inat_bird_quant_segment_1_of_3_edgetpu.tflite",
        "models/mobilenet_v2_1.0_224_inat_bird_quant_segment_2_of_3_edgetpu.tflite",
    ]
    runner = SegmentsRunner(segment_paths, device="pci:0")

    labels = read_label_file("segments_runner/test_data/inat_bird_labels.txt")

    runner.set_image(Image.open(image_path))

    # Run inference
    print("----INFERENCE TIME----")
    for _ in range(10):
        start = time.perf_counter()
        runner.invoke_all()
        inference_time = time.perf_counter() - start
        classes_dict = runner.get_result(top_n=5, detection=False)
        print("%.1fms" % (inference_time * 1000))

    print("-------RESULTS--------")
    for label, score in classes_dict.items():
        print(f"{labels.get(label, label)}: {score:.5f}")


if __name__ == "__main__":
    main()
