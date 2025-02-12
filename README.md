# SegmentsRunner

**SegmentsRunner** is a Python module that sequentially executes one or more Edge TPU models split into multiple segments. It supports both classification and detection workflows.

## Installation

### Install Dependency

1. **install tflite-runtime**
    - Example below (Python 3.9, x86_64 Linux):

        ```sh
        python3 -m pip install \
        https://github.com/google-coral/pycoral/releases/download/v2.0.0/\
        tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl
        ```

2. **install PyCoral**

    - Example below (Python 3.9, x86_64 Linux):

        ```sh
        python3 -m pip install \
        https://github.com/google-coral/pycoral/releases/download/v2.0.0/\
        pycoral-2.0.0-cp39-cp39-linux_x86_64.whl#sha256=77c81c64a99119019c0d65ae9b1af25d2856ab6057dac27d3ea64dac963bef16
        ```

### Instal Package (Pip)

```sh
pip install segments-runner
```

### Install Package (Source)

1. **Clone the Repository**  

   ```shell
   git clone https://github.com/HanChangHun/SegmentsRunner.git
   cd SegmentsRunner
   ```

2. **Install the Package**  
   Since the project uses a `pyproject.toml` file, you can install it locally with:

   ```shell
   pip install .
   ```

### Install Using `uv`

If you prefer using `uv` for dependency management, you can install the package as follows:

1. **Install Dependencies**

    ```sh
    uv pip install -r requirements.txt
    ```

2. **Install SegmentsRunner**

    ```sh
    uv pip install .
    ```

## Usage Examples

### Classification Example

Below is a quick example of running two classification model segments on a single image:

```python
from pathlib import Path
from segments_runner import SegmentsRunner

# Paths to two classification model segments and a label file
model_paths = [
    Path("model_segment1.tflite"),
    Path("model_segment2.tflite")
]
labels_path = "labels.txt"

# Initialize the runner
runner = SegmentsRunner(
    model_paths=model_paths,
    labels_path=labels_path,
    input_file="test_image.jpg"
)

# Invoke all segments
runner.invoke_all(task="classification")

# Retrieve classification results
results = runner.get_result(top_n=3)  # Top 3 classes
print("Classification Results:", results)
```

### Detection Example

For object detection, simply specify `task="detection"` when invoking and call `get_result(detection=True)`:

```python
from pathlib import Path
from segments_runner import SegmentsRunner

# Paths to two detection model segments and a COCO label file
model_paths = [
    Path("detection_segment1.tflite"),
    Path("detection_segment2.tflite")
]
labels_path = "coco_labels.txt"

# Initialize the runner
runner = SegmentsRunner(
    model_paths=model_paths,
    labels_path=labels_path,
    input_file="sample_image.jpg"
)

# Invoke all segments for detection
runner.invoke_all(task="detection")

# Retrieve detection results with a threshold
detection_results = runner.get_result(
    detection=True,
    score_threshold=0.5
)

# Print detected objects
for obj in detection_results:
    label_name = runner.labels.get(obj.id, obj.id)
    print(f"Detected {label_name} with score {obj.score:.2f} at {obj.bbox}")
```
