// Face landmark detection (478-point mesh with irises, optional blendshapes),
// mirroring mediapipe/tasks/cc/vision/face_landmarker.
//
//   auto landmarker = mlxpipe::FaceLandmarker::Create({.model_asset_path = "face_landmarker.task",
//                                                      .output_face_blendshapes = true});
//   auto result = landmarker->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mlxpipe/formats/classification.h"
#include "mlxpipe/formats/image.h"
#include "mlxpipe/formats/landmark.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"
#include "mlxpipe/model/tflite_model.h"
#include "mlxpipe/tasks/face_detector.h"

namespace mlxpipe {

// The 146 landmarks the blendshapes model reads, and its 52 output names.
extern const std::vector<int> kFaceBlendshapesLandmarkSubset;
extern const std::vector<std::string> kFaceBlendshapeNames;

struct FaceLandmarksDetectorGraphOptions {
    float min_detection_confidence = 0.5f;  // face presence threshold
};

struct FaceLandmarksDetectorOutputs {
    // Entries for rects whose face presence passed the threshold (and `keep`).
    Stream<std::vector<NormalizedLandmarkList>> landmarks;
    Stream<std::vector<NormalizedRect>> face_rects_next_frame;
    Stream<std::vector<float>> presence_scores;
};

// MultiFaceLandmarksDetectorGraph, batched over the rects. `keep` optionally
// drops rects up front (used when association replaces tracked faces).
FaceLandmarksDetectorOutputs BuildFaceLandmarksDetectorGraph(Graph& graph, Stream<Image> image,
                                                             Stream<std::vector<NormalizedRect>> face_rects,
                                                             std::shared_ptr<const TfLiteModel> model,
                                                             const FaceLandmarksDetectorGraphOptions& options,
                                                             std::optional<Stream<std::vector<bool>>> keep = {});

// FaceBlendshapesGraph: 52 blendshape scores per face.
Stream<std::vector<ClassificationList>> BuildFaceBlendshapesGraph(Graph& graph,
                                                                  Stream<std::vector<NormalizedLandmarkList>> landmarks,
                                                                  Stream<ImageSize> image_size,
                                                                  std::shared_ptr<const TfLiteModel> model);

struct FaceLandmarkerGraphOptions {
    FaceDetectorGraphOptions face_detector;
    FaceLandmarksDetectorGraphOptions face_landmarks_detector;
    float min_tracking_confidence = 0.5f;
    bool use_stream_mode = true;
    // One-euro smoothing of tracked faces (stream mode only).
    bool smooth_landmarks = true;
};

struct FaceLandmarkerOutputs {
    Stream<std::vector<NormalizedLandmarkList>> landmarks;
    Stream<std::vector<NormalizedRect>> face_rects_next_frame;
    Stream<std::vector<Detection>> detections;
    // Set when a blendshapes model is given.
    std::optional<Stream<std::vector<ClassificationList>>> blendshapes;
};

FaceLandmarkerOutputs BuildFaceLandmarkerGraph(Graph& graph, Stream<Image> image,
                                               std::shared_ptr<const TfLiteModel> detector_model,
                                               std::shared_ptr<const TfLiteModel> landmarks_model,
                                               std::shared_ptr<const TfLiteModel> blendshapes_model,
                                               const FaceLandmarkerGraphOptions& options);

struct FaceLandmarkerOptions {
    // A MediaPipe face_landmarker.task bundle.
    std::string model_asset_path;
    int num_faces = 1;
    float min_face_detection_confidence = 0.5f;
    float min_face_presence_confidence = 0.5f;
    float min_tracking_confidence = 0.5f;
    bool output_face_blendshapes = false;
    // Smooth landmarks in DetectForVideo (MediaPipe does this for one face).
    bool smooth_landmarks = true;
};

struct FaceLandmarkerResult {
    std::vector<NormalizedLandmarkList> face_landmarks;
    std::vector<ClassificationList> face_blendshapes;
};

class FaceLandmarker {
public:
    static std::unique_ptr<FaceLandmarker> Create(const FaceLandmarkerOptions& options);
    ~FaceLandmarker();

    FaceLandmarkerResult Detect(const Image& image);
    FaceLandmarkerResult DetectForVideo(const Image& image, int64_t timestamp_ms);
    int last_gpu_sync_count() const { return last_sync_count_; }

private:
    struct Pipeline;
    FaceLandmarker() = default;
    FaceLandmarkerOptions options_;
    std::shared_ptr<const TfLiteModel> detector_model_, landmarks_model_, blendshapes_model_;
    std::unique_ptr<Pipeline> image_pipeline_, video_pipeline_;
    int last_sync_count_ = 0;
};

}  // namespace mlxpipe
