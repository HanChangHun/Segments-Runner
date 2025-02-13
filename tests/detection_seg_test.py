import time

from PIL import Image, ImageDraw
from pycoral.utils.dataset import read_label_file

from segments_runner import SegmentsRunner


def draw_objects(draw, objs, labels):
    """Draw bounding boxes and labels for each detected object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline="red", width=2)
        draw.text(
            (bbox.xmin + 10, bbox.ymin + 10),
            f"{labels.get(obj.id, obj.id)}\n{obj.score:.2f}",
            fill="red",
        )


def main():
    # 모델이 세 개의 세그먼트로 나뉘어 있다고 가정
    segment_paths = [
        "models/tf2_ssd_mobilenet_v2_coco17_ptq_segment_0_of_3_edgetpu.tflite",
        "models/tf2_ssd_mobilenet_v2_coco17_ptq_segment_1_of_3_edgetpu.tflite",
        "models/tf2_ssd_mobilenet_v2_coco17_ptq_segment_2_of_3_edgetpu.tflite",
    ]

    # 입력/출력 경로
    image_path = "segments_runner/test_data/face.jpg"
    output_path = "segments_runner/test_data/det_output_seg.jpg"

    # COCO 라벨 파일
    label_path = "segments_runner/test_data/coco_labels.txt"
    labels = read_label_file(label_path)

    # SegmentsRunner 생성 시, input_file 인자로 이미지 경로를 전달하면
    # 내부에서 자동으로 모델 입력 크기에 맞춰 리사이즈를 수행합니다.
    runner = SegmentsRunner(model_paths=segment_paths)
    image = Image.open(image_path)
    runner.set_image(image, detection=True)

    print("----INFERENCE TIME----")
    # 여러 번 반복하여 평균 추론 시간 등을 확인
    for _ in range(10):
        start = time.perf_counter()
        # 감지 모델이므로 task="detection"
        runner.invoke_all(task="detection")
        inference_time = time.perf_counter() - start

        # 감지 결과를 가져옴 (score_threshold=0.4)
        objs = runner.get_result(detection=True)
        print(f"{inference_time * 1000:.2f}ms")

    print("-------RESULTS--------")
    if not objs:
        print("No objects detected")
    else:
        for obj in objs:
            label_name = labels.get(obj.id, obj.id)
            print(f"Label: {label_name}")
            print(f"  ID:    {obj.id}")
            print(f"  Score: {obj.score:.5f}")
            print(f"  BBox:  {obj.bbox}")

    # 시각화(바운딩 박스 그리기)
    image = Image.open(image_path).convert("RGB")
    draw_objects(ImageDraw.Draw(image), objs, labels)
    image.save(output_path)
    print(f"Detection result saved to {output_path}")


if __name__ == "__main__":
    main()
