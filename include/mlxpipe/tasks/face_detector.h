// Face detection, mirroring mediapipe/tasks/cc/vision/face_detector
// (BlazeFace short-range, 128x128).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlxpipe/formats/detection.h"
#include "mlxpipe/formats/image.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"
#include "mlxpipe/model/tflite_model.h"

namespace mlxpipe {

struct FaceDetectorGraphOptions {
    float min_detection_confidence = 0.5f;
    float min_suppression_threshold = 0.3f;
    int num_faces = 1;
};

struct FaceDetectorOutputs {
    // Normalized to the input image; keypoints: right eye, left eye, nose tip,
    // mouth centre, right ear tragion, left ear tragion.
    Stream<std::vector<Detection>> detections;
    Stream<std::vector<NormalizedRect>> face_rects;
    // face_rects scaled by 1.5: the regions the landmark model expects.
    Stream<std::vector<NormalizedRect>> expanded_face_rects;
};

// FaceDetectorGraph.
FaceDetectorOutputs BuildFaceDetectorGraph(Graph& graph, Stream<Image> image,
                                           std::shared_ptr<const TfLiteModel> model,
                                           const FaceDetectorGraphOptions& options);

struct FaceDetectorOptions {
    // face_detector.tflite (or a face_landmarker.task bundle).
    std::string model_asset_path;
    float min_detection_confidence = 0.5f;
    float min_suppression_threshold = 0.3f;
    int num_faces = -1;  // -1: no limit
};

struct FaceDetectorResult {
    std::vector<Detection> detections;
};

class FaceDetector {
public:
    static std::unique_ptr<FaceDetector> Create(const FaceDetectorOptions& options);
    FaceDetectorResult Detect(const Image& image);
    FaceDetectorResult DetectForVideo(const Image& image, int64_t timestamp_ms);

private:
    FaceDetector() = default;
    Graph graph_;
    GraphInput<Image> image_;
    GraphOutput<std::vector<Detection>> detections_;
    int64_t last_timestamp_ms_ = -1;
};

}  // namespace mlxpipe
