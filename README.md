# MediaPipe MLX Port

A native C++ reimplementation of Google MediaPipe's **hand-tracking pipeline** (palm detection + 21-point hand landmarks) that runs entirely on [Apple's MLX](https://github.com/ml-explore/mlx) array framework, using the GPU via Metal instead of TensorFlow Lite. It reproduces MediaPipe's detector/tracker logic (ROI cropping, rotation normalization, per-hand tracking, One Euro Filter smoothing, IoU-based deduplication) directly in C++, so the original MediaPipe/TFLite runtime is not needed at inference time.

## How it works

MediaPipe's hand solution ships two TFLite models — a palm detector (BlazePalm) and a hand landmark model (BlazeHand) — wired together by a MediaPipe graph that handles cropping, rotation, tracking, and smoothing. This project ports both halves to MLX:

1. **Model conversion** (`tools/`) — offline scripts read the original `.tflite` FlatBuffer files and produce two things per model:
   - `tools/convert_tflite_to_safetensors.py` extracts the trained weight tensors into a `.safetensors` file.
   - `tools/transpile_tflite_to_mlx.py` walks the TFLite op graph and generates equivalent MLX C++ (`mlx::core::conv2d`, `pad`, `add`, PReLU/ReLU6 activations, etc.) as a `forward()` method.

   The generated inference graphs live in [src/blaze_palm.h](src/blaze_palm.h) (`BlazePalm`) and [src/blaze_hand.h](src/blaze_hand.h) (`BlazeHand`); the corresponding weights are the `.safetensors` files in [tools/models/](tools/models).

2. **Runtime pipeline** ([src/vision.cpp](src/vision.cpp)) — `detect_hands()` reimplements MediaPipe's hand-tracking graph logic per frame:
   - Pads/resizes the frame to 192×192 for palm detection.
   - For each already-tracked hand, recomputes its rotated crop from the previous frame's landmarks (MediaPipe's rotation-normalization math), runs `BlazeHand`, and smooths the result with a [One Euro Filter](src/one_euro_filter.cpp).
   - Runs `BlazePalm` + SSD anchor decoding ([src/ssd_anchors.cpp](src/ssd_anchors.cpp), [src/tensors_to_detections.cpp](src/tensors_to_detections.cpp)) and non-max suppression ([src/non_max_suppression.cpp](src/non_max_suppression.cpp)) whenever fewer than `MAX_HANDS` are actively tracked, to (re)acquire hands.
   - Uses IoU between tracked ROIs and new palm detections to avoid spawning duplicate trackers for the same hand.
   - Returns per-hand bounding boxes, 21 2D/3D landmarks, and confidence scores.

3. **Demo app** ([main.cpp](main.cpp)) — opens the webcam with OpenCV, calls `mlx_vision::detect_hands()` per frame, and draws the bounding box, landmark points, and finger-skeleton connections with an FPS/inference-time overlay.

## Layout

| Path | Contents |
|---|---|
| [`src/`](src) | The `mlx_vision` library: pipeline orchestration, transpiled BlazePalm/BlazeHand MLX graphs, anchors, NMS, tensor decoding, One Euro filter |
| [`tools/`](tools) | TFLite → safetensors / TFLite → MLX-C++ conversion scripts, plus the source `.tflite` models and generated `.safetensors` weights |
| [`main.cpp`](main.cpp) | Standalone webcam demo binary (`test_app`) |
| [`third_party/mlx.BUILD`](third_party/mlx.BUILD) | Bazel build rule for a locally-installed MLX C++ build |
| [`mediapipe/`](mediapipe) | Vendored copy of the upstream [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) repo, kept for reference to the original graphs/models being ported |

## Build

This is a macOS/Apple Silicon project built with [Bazel](https://bazel.build), and expects OpenCV and MLX to already be installed locally (paths are wired in [`MODULE.bazel`](MODULE.bazel) to Homebrew's `opencv` and a local `/usr/local` MLX C++ build):

```bash
bazel build //:test_app
```

Run the webcam demo:

```bash
bazel-bin/test_app
```

## Status

This is an experimental, in-progress port — the transpiled model code is machine-generated (hence the unwieldy operator names in `blaze_palm.h`/`blaze_hand.h`), and only the hand-tracking solution has been ported so far.
