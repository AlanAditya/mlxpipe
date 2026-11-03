# mlxpipe

MediaPipe's vision pipelines, rebuilt for Apple Silicon on [MLX](https://github.com/ml-explore/mlx).

mlxpipe runs MediaPipe's own models (`.task` bundles and `.tflite` files, loaded directly) on the GPU through MLX. It re-implements MediaPipe's calculators one-to-one, so every stage can be compared against the original. Pipelines are assembled in plain C++: no `.pbtxt` graph files, no protobuf options, no Bazel requirement.

**Included tasks**

| Task | Models | Output |
|---|---|---|
| `HandLandmarker` | `hand_landmarker.task` | 21 landmarks per hand (image and world), handedness |
| `FaceLandmarker` | `face_landmarker.task` | 478 landmarks per face (mesh + irises), optional 52 blendshapes |
| `FaceDetector` | `face_landmarker.task` or `face_detector.tflite` | face boxes + 6 keypoints |

## Quick start

```cpp
#include <mlxpipe/mlxpipe.h>

auto hands = mlxpipe::HandLandmarker::Create({
    .model_asset_path = "tools/models/hand_landmarker.task",
    .num_hands = 2,
});

cv::Mat frame = ...;  // BGR, e.g. from cv::VideoCapture
auto result = hands->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
for (size_t i = 0; i < result.hand_landmarks.size(); ++i) {
    const auto& wrist = result.hand_landmarks[i][0];          // normalized x, y, z
    const auto& label = result.handedness[i][0].label;        // "Left" / "Right"
}
```

`Detect(image)` treats every call as an independent image (palm/face detection every time). `DetectForVideo(image, timestamp_ms)` tracks across frames and only runs the detector when fewer than `num_hands` / `num_faces` are tracked, like MediaPipe's VIDEO and LIVE_STREAM modes.

```cpp
auto faces = mlxpipe::FaceLandmarker::Create({
    .model_asset_path = "tools/models/face_landmarker.task",
    .num_faces = 1,
    .output_face_blendshapes = true,
});
auto r = faces->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
// r.face_landmarks[0]: 478 points; r.face_blendshapes[0]: {label, score} x 52
```

Handedness follows MediaPipe's convention, which assumes a mirrored (selfie) image; the examples flip the camera frame before detection.

## Building

Requirements: macOS on Apple Silicon, MLX (C++ library, found through CMake's `MLXConfig.cmake`), OpenCV (Homebrew `opencv`, version 5).

```bash
cmake -S . -B cmake-build
```

```bash
cmake --build cmake-build -j8
```

```bash
ctest --test-dir cmake-build
```

Install and use from another CMake project:

```bash
cmake --install cmake-build --prefix /usr/local
```

```cmake
find_package(mlxpipe REQUIRED)
target_link_libraries(my_app PRIVATE mlxpipe::mlxpipe)
```

### Face detection as a dylib

`libmlxpipe_face.dylib` (built with the library) packages face detection and face landmarks behind a plain C header, [`include/mlxpipe_face.h`](include/mlxpipe_face.h). Callers need only that header and the dylib, no MLX or OpenCV headers, so it can be used from C, C++, Objective-C, Swift (via a bridging header) or Python `ctypes`. Only the `mlxpipe_*` functions are exported.

```c
#include "mlxpipe_face.h"

mlxpipe_face_landmarker_options options;
mlxpipe_face_landmarker_default_options(&options);
options.model_path = "tools/models/face_landmarker.task";
options.output_blendshapes = 1;
mlxpipe_face_landmarker* faces = mlxpipe_face_landmarker_create(&options);
if (!faces) fprintf(stderr, "%s\n", mlxpipe_last_error());

// e.g. a locked kCVPixelFormatType_32BGRA CVPixelBuffer
mlxpipe_image image = {base_address, width, height, bytes_per_row, MLXPIPE_PIXEL_BGRA};
mlxpipe_face_landmarks result[1];
int n = mlxpipe_face_landmarker_detect_for_video(faces, &image, timestamp_ms, result, 1);
// result[0].landmarks[1] is the nose tip; result[0].blendshapes[i] is named by mlxpipe_face_blendshape_name(i)

mlxpipe_face_landmarker_destroy(faces);
```

Each face also carries the bounding box of its landmarks (`xmin`, `ymin`, `width`, `height`). `mlxpipe_face_region_indices(MLXPIPE_FACE_REGION_LEFT_EYE, &indices)` returns the mesh indices of a feature (eyes, eyebrows, irises, nose, lips, face oval, from MediaPipe's connection tables; left and right are the subject's), so callers don't hard-code mesh indices. `mlxpipe_face_detector_*` gives just the face boxes and 6 keypoints. Images can be RGB, BGR, RGBA or BGRA with any row stride, and are read only during the call. Link with `-lmlxpipe_face` and an rpath to the dylib's folder. `cmake --install` puts the dylib in `lib/` and the header in `include/`, and CMake projects can use `mlxpipe::mlxpipe_face`. At runtime the dylib loads MLX (`libmlx.dylib`, with `mlx.metallib` next to it) and Homebrew's OpenCV `core`, `imgproc` and `geometry` libraries from where they were installed at build time.

Bazel works too (`MODULE.bazel` points at Homebrew OpenCV and an MLX install in `/usr/local`):

```bash
bazel build //:mlxpipe
```

```bash
bazel test //:all
```

## Examples

Run from the repository root (the camera needs your terminal to have camera permission):

```bash
./cmake-build/hand_tracking --max 2
```

```bash
./cmake-build/face_landmarks --blendshapes
```

```bash
./cmake-build/hands_and_face
```

Options: `--camera N`, `--max N` (hands/faces), `--no-mirror`, `--record DIR` (save every 5th raw frame for offline testing), `--frames N` (stop after N frames). The overlay shows FPS, detection time and the number of GPU synchronisations per frame.

## Performance

Measured on an Apple M4 Max, 1920x1080 webcam frames, camera-paced (≈30 fps), median detection time per frame:

| Pipeline | Median | Notes |
|---|---|---|
| Hands, 2 tracked | 8–9 ms | one GPU sync per frame |
| Hands, palm search only | 5.6 ms | no hands in view |
| Face mesh, 1 tracked | 2.4–4 ms | |
| Face detector only | 3.4 ms | no face in view |
| Hands + face in one graph | 10–12 ms | `hands_and_face`: palm detection, hand landmarks and face mesh in one GPU submission |

Back-to-back (unpaced) replay of recorded footage: hands 4.2 ms, face 2.4 ms. The gap to the paced numbers is the GPU and CPU clocking down while waiting for the next camera frame; mlxpipe minimises the number of waits instead of fighting the clocks.

### Against MediaPipe

Google's MediaPipe 1.1.1 (Python Tasks API, `HandLandmarker` with `num_hands = 2` and `FaceLandmarker`, VIDEO mode, same `.task` bundles) and mlxpipe on the same 450 consecutive 1920x1080 webcam frames (face in view throughout, two hands in 438 frames), M4 Max, hands then face per frame. Distances are the mean per-landmark distance in pixels between paired hands/faces.

| | Hand count agrees | Hand landmarks | Face count agrees | Face landmarks |
|---|---|---|---|---|
| mlxpipe vs MediaPipe CPU | 450/450 frames | 0.83 px (median 0.52, p95 2.6) | 450/450 | 0.12 px (median 0.08, p95 0.24) |
| MediaPipe GPU vs MediaPipe CPU | 50/450 | 9.2 px | 447/450 | 2.6 px |

Median time per frame, frames decoded in advance. "Back to back" runs each frame as soon as the previous one is done; "30 fps" starts frames 33 ms apart, like a camera:

| | Hands (2 tracked) | Face (1 tracked) | Hands + face |
|---|---|---|---|
| mlxpipe, back to back | 3.3 ms | 1.8 ms | 4.7 ms (one graph), 5.1 ms (two tasks) |
| MediaPipe CPU (XNNPACK), back to back | 5.5 ms | 3.3 ms | 8.8 ms |
| MediaPipe GPU (Metal), back to back | 11.5 ms | 5.6 ms | |
| mlxpipe, 30 fps | 10.2 ms | 7.5 ms | 11.5 ms (one graph) |
| MediaPipe CPU, 30 fps | 7.1 ms | 3.9 ms | 11.0 ms |

The models alone (median per call, warm):

| Model, input | MLX (GPU) | XNNPACK, 1 thread | XNNPACK, 4 threads |
|---|---|---|---|
| Hand landmarks 224, two hands | 3.05 ms (one batch of 2) | 5.16 ms (2 x 2.58) | 2.94 ms (2 x 1.47) |
| Palm detector 192 | 1.43 ms | 4.76 ms | 2.44 ms |
| Face mesh 256 | 1.63 ms | 2.32 ms | 1.83 ms |
| Face detector 128 | 0.74 ms | 0.56 ms | 0.64 ms |

- **Accuracy:** mlxpipe finds the same number of hands and faces as MediaPipe's CPU path in every frame, and its landmarks are within about a pixel. The residual comes from floating-point differences between MLX's GPU kernels and XNNPACK's CPU kernels, which tracking then carries forward (each frame's hand ROI comes from the previous frame's landmarks). Faces differ less, likely because both sides run the One Euro filter over them. None of the intentional differences below shows up as a gap on this footage, including the knuckle-subset hand ROI: MediaPipe never lost a hand that mlxpipe kept. MediaPipe's own GPU delegate differs from its CPU path far more: it drops the second hand in most two-hand frames.
- **Back to back, mlxpipe is faster:** 1.7x for hands, 1.8x for the face and 1.9x for both in one graph, compared with MediaPipe's CPU path (3.5x and 3.1x compared with its GPU delegate). MediaPipe runs XNNPACK with one thread on desktop (`GetCpuDefaultNumThreads`), so its hand time is about two single-threaded landmark-model calls. In mlxpipe, the single GPU evaluation is over 90% of a hand frame, and most of that is the batched landmark model. The CPU-side nodes (crops, decoding, projections, rects) add up to about 0.3 ms.
- **At 30 fps, MediaPipe's CPU path is faster:** 1.4x for hands and 1.9x for the face. The one-graph hands + face pipeline roughly ties with MediaPipe running both tasks (11.5 vs 11.0 ms). The cause is the GPU clocking down in the ~25 ms between frames. Keeping the GPU continuously busy between frames brings the 30 fps numbers back to the back-to-back ones (3.3 / 1.9 / 4.7 ms). Lighter workarounds do not: a tiny GPU eval every 1 ms, or warming the GPU for the last 3–10 ms before a frame, change nothing. Keeping only the CPU busy brings the CPU-side nodes back to speed (hands 7.3 ms) but not the GPU wait. A higher thread QoS does not help either.

To reproduce the accuracy comparison: run MediaPipe's Tasks API in VIDEO mode over the same frames, with timestamps advancing 33 ms per frame. Write each frame as one line (count, then `x y` for every landmark) and compare with `mlxpipe_replay frames/ --compare hands.txt --compare-faces faces.txt`. `mlxpipe_replay`'s own timings are back to back, but with `--overlays` the PNG encoding between frames lets the GPU clock down, which roughly doubles them.

Where the speed comes from:

- **One GPU round trip per frame.** MLX builds computations lazily. The graph runs every node that only builds tensor work first, then evaluates everything pending in a single `mx::eval` before any node reads results on the CPU. Independent branches (palm detection and tracked-hand landmarks; hands and face) share one submission automatically.
- **Batched models.** Where MediaPipe loops over hands or faces one at a time, mlxpipe crops all regions into one `[N, H, W, 3]` batch and runs the model once.
- **Compiled models.** Each model's forward pass is wrapped in `mx::compile`, fusing the elementwise ops (padding, bias, activations) between convolutions. Constant subgraphs (e.g. float16 weight dequantization) are folded and weights re-laid-out once at load time.

## The graph API

Pipelines are graphs of nodes (MediaPipe's calculators) connected by typed streams, built in code:

```cpp
using namespace mlxpipe;

Graph graph;
GraphInput<Image> image = graph.input<Image>("image");

auto& to_tensor = graph.add<ImageToTensorNode>(ImageToTensorOptions{
    .output_tensor_width = 192, .output_tensor_height = 192, .keep_aspect_ratio = true});
image >> to_tensor.image;

auto& inference = graph.add<InferenceNode>(TfLiteModel::load("palm_detection.tflite"));
to_tensor.tensor >> inference.tensor;

GraphOutput<std::vector<Tensor>> outputs = graph.output(inference.tensors);

image.send(Image(frame));
graph.run(timestamp_us);
const std::vector<Tensor>& raw = outputs.get();
```

- **Ports are typed.** Connecting a `Stream<Image>` to an `Input<Tensor>` is a compile error.
- **One tick per `run()`.** Each stream carries at most one value per tick. An empty stream is MediaPipe's empty packet: a node whose required input is empty is skipped, so gating propagates.
- **Back edges** use `graph.loopback<T>(initial)`: read last tick's value from `.previous`, feed this tick's into `.next`.
- **Writing a node:** subclass `Node`, declare ports as members, implement `process()`. Override `reads_tensors()` to return true if `process()` reads tensor data on the CPU; the graph evaluates pending GPU work before running it.

```cpp
struct ScoreFilterNode : Node {
    ScoreFilterNode(float min_score) : Node("ScoreFilter"), min_score(min_score) {}
    Input<std::vector<Detection>> in{this, "in"};
    Output<std::vector<Detection>> out{this, "out"};
    float min_score;
    void process() override {
        std::vector<Detection> kept;
        for (const auto& d : in.get()) if (d.score >= min_score) kept.push_back(d);
        out.send(std::move(kept));
    }
};
```

MediaPipe's subgraphs are public builder functions that add nodes to your graph and return their output streams, so pipelines can be combined or rewired. `examples/hands_and_face.cpp` puts the hand and face pipelines in one graph:

```cpp
HandLandmarkerOutputs hands = BuildHandLandmarkerGraph(graph, image, palm_model, hand_model, hand_options);
FaceLandmarkerOutputs faces = BuildFaceLandmarkerGraph(graph, image, face_detector, face_mesh, nullptr, face_options);
```

Available builders: `BuildHandDetectorGraph`, `BuildHandLandmarksDetectorGraph`, `BuildHandLandmarkerGraph`, `BuildFaceDetectorGraph`, `BuildFaceLandmarksDetectorGraph`, `BuildFaceBlendshapesGraph`, `BuildFaceLandmarkerGraph`.

## Running models directly

```cpp
auto model = mlxpipe::TfLiteModel::load("model.tflite");                 // or from_buffer(...)
auto bundle = mlxpipe::ModelBundle::open("face_landmarker.task");        // .task = stored zip
auto mesh = mlxpipe::TfLiteModel::from_buffer(bundle.file("face_landmarks_detector.tflite"));
std::vector<mlx::core::array> out = (*mesh)({input_nhwc});               // lazy; batch dim is free
auto [lo, hi] = mesh->input_range();                                      // from the model's metadata
```

The runtime reads the TFLite FlatBuffer and its metadata itself (no TensorFlow dependency). Supported operators: `CONV_2D`, `DEPTHWISE_CONV_2D`, `FULLY_CONNECTED`, `ADD`, `SUB`, `MUL`, `DIV`, `SQUARED_DIFFERENCE`, `PRELU`, `RELU`, `RELU6`, `LOGISTIC`, `TANH`, `NEG`, `SQRT`, `RSQRT`, `MAX_POOL_2D`, `AVERAGE_POOL_2D` (valid padding), `PAD`, `RESHAPE`, `CONCATENATION`, `RESIZE_BILINEAR`, `MEAN`, `SUM`, `TRANSPOSE`, `STRIDED_SLICE`, `DEQUANTIZE`. Loading a model with other operators fails with a message naming the operator.

## Mapping to MediaPipe

Each node ports one MediaPipe calculator; option fields keep MediaPipe's names.

| mlxpipe | MediaPipe source |
|---|---|
| `ImageToTensorNode`, `GetRoi`, `PadRoi`, `GetRotatedSubRectToRectTransformMatrix` | `calculators/tensor/image_to_tensor_calculator.cc`, `image_to_tensor_utils.cc`, `image_to_tensor_converter_opencv.cc` |
| `InferenceNode`, `TfLiteModel` | `calculators/tensor/inference_calculator.cc` (TFLite interpreter) |
| `GenerateSsdAnchors` | `calculators/tflite/ssd_anchors_calculator.cc` |
| `TensorsToDetectionsNode` | `calculators/tensor/tensors_to_detections_calculator.cc` |
| `NonMaxSuppressionNode` | `calculators/util/non_max_suppression_calculator.cc` |
| `DetectionProjectionNode` | `calculators/util/detection_projection_calculator.cc` |
| `DetectionsToRectsNode` | `calculators/util/detections_to_rects_calculator.cc` |
| `RectTransformationNode` | `calculators/util/rect_transformation_calculator.cc` |
| `TensorsToLandmarksNode` | `calculators/tensor/tensors_to_landmarks_calculator.cc` |
| `TensorsToFloatsNode`, `ThresholdingNode` | `calculators/tensor/tensors_to_floats_calculator.cc`, `calculators/util/thresholding_calculator.cc` |
| `TensorsToClassificationNode` | `calculators/tensor/tensors_to_classification_calculator.cc` |
| `LandmarkLetterboxRemovalNode`, `LandmarkProjectionNode`, `WorldLandmarkProjectionNode` | `calculators/util/landmark_*`, `world_landmark_projection_calculator.cc` |
| `LandmarksToDetectionNode`, `LandmarksToTensorNode` | `calculators/util/landmarks_to_detection_calculator.cc`, `calculators/tensor/landmarks_to_tensor_calculator.cc` |
| `LandmarksSmoothingNode` | `calculators/util/landmarks_smoothing_calculator.cc`, `util/filtering/one_euro_filter.cc` |
| `HandLandmarksToRectNode` | `modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc` |
| `HandAssociationNode`, `HandLandmarksDeduplicationNode` | `tasks/cc/vision/hand_landmarker/calculators/` |
| `AssociationNormRectNode` | `calculators/util/association_norm_rect_calculator.cc` |
| `GateNode`, `HasMinSizeNode`, `ClipVectorSizeNode`, `ConcatenateVectorNode`, `LoopbackNode` | `calculators/core/` (gate, collection_has_min_size, clip_vector_size, concatenate_vector, previous_loopback) |

Task graphs follow `tasks/cc/vision/{hand_detector,hand_landmarker,face_detector,face_landmarker}` and take their constants (anchors, decoding, thresholds, rect expansion) from those files. Input value ranges come from each model's TFLite metadata, as MediaPipe's `ImagePreprocessingGraph` does.

## Intentional differences from MediaPipe

- **Batching instead of loops.** Per-hand and per-face subgraphs run as one batch. Results are the same; `FilterByFlagsNode` stands in for MediaPipe's per-item `AllowIf` gates.
- **Tracked regions run alongside detection.** MediaPipe runs landmarks on the associated rects after detection. Association keeps (hands) or only replaces (faces) tracked rects, so mlxpipe runs landmarks on tracked rects in the same GPU submission as detection and only processes newly found rects afterwards. Faces replaced by a fresh detection are dropped, matching `AssociationNormRectCalculator`.
- **Hand ROI from the knuckle landmarks.** `HandLandmarksToRectCalculator` indexes joints 4, 6 and 8, which refer to a 12-landmark subset (wrist, thumb base, each finger's MCP and PIP). The vendored Tasks graph passes all 21 landmarks, which turns those into thumb tip / index PIP / index tip. On recorded webcam footage the full list lost the second hand in 34 of 123 two-hand frames, so mlxpipe uses the subset.
- **Smoothing per tracked face.** MediaPipe only smooths when `num_faces == 1`. mlxpipe keeps a One Euro filter per tracked face id (same filter parameters), so smoothing also works with several faces.
- **No zero rect for empty detections.** `DetectionsToRects` returns no rects when there are no detections, instead of one zero-sized rect.
- **Loopbacks fall back to their initial value** after a tick in which nothing was sent, so "nothing tracked" restarts detection without extra gating logic.
- **Not implemented yet:** face geometry (facial transformation matrices), the long-range face detector, pose, segmentation and other MediaPipe tasks.

## Testing

- `graph_test`: wiring, gating, loopbacks, single GPU sync for independent branches.
- `model_golden_test`: every model in the `.task` bundles against golden outputs (`tests/data/golden`), compiled vs eager, batch vs single. Regenerate after an intended change with `MLXPIPE_UPDATE_GOLDEN=1`.
- `calculators_test`: calculator math against MediaPipe's conventions (rotation, rect transformation, projection, letterboxing, NMS, deduplication).
- `hand_landmarker_test`, `face_landmarker_test`: end-to-end smoke tests and paced timing.
- `c_api_test`: the face C API, compiled as C99 against `libmlxpipe_face.dylib` alone.

Real-footage regression check: record frames with an example's `--record DIR`, save a baseline, and compare later builds against it:

```bash
./cmake-build/mlxpipe_replay frames/ --save baseline.txt --save-faces baseline_faces.txt --overlays overlays/
```

```bash
./cmake-build/mlxpipe_replay frames/ --compare baseline.txt --compare-faces baseline_faces.txt
```

## Layout

| Path | Contents |
|---|---|
| `include/mlxpipe/framework` | `Graph`, `Node`, ports, streams, loopbacks |
| `include/mlxpipe/formats` | `Image`, `NormalizedRect`, `Detection`, landmarks, classifications (mirroring MediaPipe's protos) |
| `include/mlxpipe/calculators` | the calculator nodes |
| `include/mlxpipe/model` | `TfLiteModel`, `ModelBundle` |
| `include/mlxpipe/tasks` | task APIs, subgraph builders, landmark connection tables |
| `include/mlxpipe_face.h`, `src/c_api` | the face C API of `libmlxpipe_face.dylib` |
| `src/` | implementation, same structure |
| `examples/` | webcam demos |
| `tests/` | unit, golden and end-to-end tests |
| `tools/models` | MediaPipe model bundles |
| `tools/replay` | offline replay and comparison tool |
| `mediapipe/` | vendored MediaPipe checkout (reference only, not built) |

MediaPipe's models and the ported calculator logic are Apache 2.0 licensed by Google.
