// Hand landmark detection, mirroring mediapipe/tasks/cc/vision/hand_landmarker.
//
// Quick start:
//   auto landmarker = mlxpipe::HandLandmarker::Create({.model_asset_path = "hand_landmarker.task",
//                                                      .num_hands = 2});
//   auto result = landmarker->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
//
// The subgraph builders are public so the pipeline can be extended or
// rewired: each returns the streams its MediaPipe counterpart outputs.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlxpipe/formats/classification.h"
#include "mlxpipe/formats/detection.h"
#include "mlxpipe/formats/image.h"
#include "mlxpipe/formats/landmark.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"
#include "mlxpipe/model/tflite_model.h"

namespace mlxpipe {

// --- Subgraphs ---------------------------------------------------------------

struct HandDetectorGraphOptions {
    float min_detection_confidence = 0.5f;
    int num_hands = 1;
};

struct HandDetectorOutputs {
    Stream<std::vector<Detection>> palm_detections;
    Stream<std::vector<NormalizedRect>> palm_rects;
    Stream<std::vector<NormalizedRect>> hand_rects;
};

// HandDetectorGraph: palm detection and palm-to-hand-rect conversion.
HandDetectorOutputs BuildHandDetectorGraph(Graph& graph, Stream<Image> image,
                                           std::shared_ptr<const TfLiteModel> palm_model,
                                           const HandDetectorGraphOptions& options);

struct HandLandmarksDetectorGraphOptions {
    float min_detection_confidence = 0.5;  // hand presence threshold
};

struct HandLandmarksDetectorOutputs {
    // One entry per input rect whose hand presence passed the threshold.
    Stream<std::vector<NormalizedLandmarkList>> landmarks;
    Stream<std::vector<LandmarkList>> world_landmarks;
    Stream<std::vector<ClassificationList>> handedness;
    Stream<std::vector<NormalizedRect>> hand_rects_next_frame;
    Stream<std::vector<float>> presence_scores;
};

// MultipleHandLandmarksDetectorGraph, batched: all rects go through the model
// in a single [N, 224, 224, 3] invocation.
HandLandmarksDetectorOutputs BuildHandLandmarksDetectorGraph(Graph& graph, Stream<Image> image,
                                                             Stream<std::vector<NormalizedRect>> hand_rects,
                                                             std::shared_ptr<const TfLiteModel> landmarks_model,
                                                             const HandLandmarksDetectorGraphOptions& options);

struct HandLandmarkerGraphOptions {
    HandDetectorGraphOptions hand_detector;
    HandLandmarksDetectorGraphOptions hand_landmarks_detector;
    float min_tracking_confidence = 0.5f;
    // Track hands across frames (VIDEO/LIVE_STREAM) instead of detecting
    // palms on every image (IMAGE).
    bool use_stream_mode = true;
};

struct HandLandmarkerOutputs {
    Stream<std::vector<NormalizedLandmarkList>> landmarks;
    Stream<std::vector<LandmarkList>> world_landmarks;
    Stream<std::vector<ClassificationList>> handedness;
    Stream<std::vector<NormalizedRect>> hand_rects_next_frame;
    Stream<std::vector<Detection>> palm_detections;
};

// HandLandmarkerGraph. In stream mode, landmarks for tracked hands run in the
// same GPU submission as palm detection; only newly found hands need a second
// landmark pass.
HandLandmarkerOutputs BuildHandLandmarkerGraph(Graph& graph, Stream<Image> image,
                                               std::shared_ptr<const TfLiteModel> palm_model,
                                               std::shared_ptr<const TfLiteModel> landmarks_model,
                                               const HandLandmarkerGraphOptions& options);

// --- Task API ----------------------------------------------------------------

struct HandLandmarkerOptions {
    // A MediaPipe hand_landmarker.task bundle...
    std::string model_asset_path;
    // ...or the two models directly.
    std::string hand_detector_model_path;
    std::string hand_landmarks_model_path;

    int num_hands = 1;
    float min_hand_detection_confidence = 0.5f;
    float min_hand_presence_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
};

struct HandLandmarkerResult {
    std::vector<ClassificationList> handedness;
    std::vector<NormalizedLandmarkList> hand_landmarks;
    std::vector<LandmarkList> hand_world_landmarks;
};

class HandLandmarker {
public:
    // Throws std::runtime_error if the models cannot be loaded.
    static std::unique_ptr<HandLandmarker> Create(const HandLandmarkerOptions& options);

    // Detects hands in an independent image (palm detection every call).
    HandLandmarkerResult Detect(const Image& image);
    // Tracks hands across video frames; timestamps must increase.
    HandLandmarkerResult DetectForVideo(const Image& image, int64_t timestamp_ms);

    // Number of GPU synchronisations in the last call.
    int last_gpu_sync_count() const;

private:
    struct Pipeline;
    HandLandmarker();
    std::unique_ptr<Pipeline> image_pipeline_;
    std::unique_ptr<Pipeline> video_pipeline_;
    HandLandmarkerOptions options_;
    std::shared_ptr<const TfLiteModel> palm_model_;
    std::shared_ptr<const TfLiteModel> landmarks_model_;
    int last_sync_count_ = 0;

public:
    ~HandLandmarker();
};

}  // namespace mlxpipe
