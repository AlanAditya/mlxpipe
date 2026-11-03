// Landmark calculators, batched over the items of a model batch:
//   TensorsToLandmarks      mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc
//   TensorsToFloats         mediapipe/calculators/tensor/tensors_to_floats_calculator.cc
//   Thresholding            mediapipe/calculators/util/thresholding_calculator.cc
//   TensorsToClassification mediapipe/calculators/tensor/tensors_to_classification_calculator.cc
//   LandmarkLetterboxRemoval, LandmarkProjection, WorldLandmarkProjection
//                           mediapipe/calculators/util/
#pragma once

#include <string>
#include <vector>

#include "mlxpipe/formats/classification.h"
#include "mlxpipe/formats/detection.h"
#include "mlxpipe/formats/landmark.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct TensorsToLandmarksOptions {
    enum class Activation { kNone, kSigmoid };

    int num_landmarks = 0;
    // Model input size, used to normalize coordinates.
    int input_image_width = 0;
    int input_image_height = 0;
    bool flip_horizontally = false;
    bool flip_vertically = false;
    float normalize_z = 1.0f;
    Activation visibility_activation = Activation::kNone;
    Activation presence_activation = Activation::kNone;
    // Which model output holds the landmarks.
    int tensor_index = 0;
};

// Decodes item `item` of a [N, num_landmarks * D] tensor (D >= 2).
LandmarkList DecodeLandmarks(const float* data, int num_values, const TensorsToLandmarksOptions& options);
NormalizedLandmarkList NormalizeLandmarks(const LandmarkList& landmarks, const TensorsToLandmarksOptions& options);

class TensorsToLandmarksNode : public Node {
public:
    explicit TensorsToLandmarksNode(TensorsToLandmarksOptions options)
        : Node("TensorsToLandmarks"), options_(options) {}

    Input<std::vector<Tensor>> tensors{this, "tensors"};
    // Coordinates normalized by the model input size.
    Output<std::vector<NormalizedLandmarkList>> normalized_landmarks{this, "normalized_landmarks"};
    // Raw model values (e.g. world landmarks in metres).
    Output<std::vector<LandmarkList>> landmarks{this, "landmarks"};

    bool reads_tensors() const override { return true; }
    void process() override;

private:
    TensorsToLandmarksOptions options_;
};

// The first value of each batch item of one model output, optionally passed
// through a sigmoid (TensorsToFloatsCalculatorOptions.activation).
class TensorsToFloatsNode : public Node {
public:
    explicit TensorsToFloatsNode(int tensor_index, bool sigmoid = false)
        : Node("TensorsToFloats"), tensor_index_(tensor_index), sigmoid_(sigmoid) {}

    Input<std::vector<Tensor>> tensors{this, "tensors"};
    Output<std::vector<float>> floats{this, "floats"};

    bool reads_tensors() const override { return true; }
    void process() override;

private:
    int tensor_index_;
    bool sigmoid_;
};

class ThresholdingNode : public Node {
public:
    explicit ThresholdingNode(float threshold) : Node("Thresholding"), threshold_(threshold) {}

    Input<std::vector<float>> values{this, "values"};
    Output<std::vector<bool>> flags{this, "flags"};

    void process() override;

private:
    float threshold_;
};

struct TensorsToClassificationOptions {
    // A single sigmoid output scoring label_map[0] (with 1 - score for label_map[1]).
    bool binary_classification = false;
    std::vector<std::string> label_map;
    int top_k = 0;
    int tensor_index = 0;
};

class TensorsToClassificationNode : public Node {
public:
    explicit TensorsToClassificationNode(TensorsToClassificationOptions options)
        : Node("TensorsToClassification"), options_(std::move(options)) {}

    Input<std::vector<Tensor>> tensors{this, "tensors"};
    Output<std::vector<ClassificationList>> classifications{this, "classifications"};

    bool reads_tensors() const override { return true; }
    void process() override;

private:
    TensorsToClassificationOptions options_;
};

NormalizedLandmarkList RemoveLetterbox(const NormalizedLandmarkList& landmarks, const LetterboxPadding& padding);
// NORM_RECT mode of LandmarkProjectionCalculator (square ROIs).
NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const NormalizedRect& rect,
                                        bool ignore_rotation = false);
// PROJECTION_MATRIX mode of LandmarkProjectionCalculator.
NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const Matrix4& matrix);
LandmarkList ProjectWorldLandmarks(const LandmarkList& landmarks, const NormalizedRect& rect);

class LandmarkLetterboxRemovalNode : public Node {
public:
    LandmarkLetterboxRemovalNode() : Node("LandmarkLetterboxRemoval") {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<std::vector<LetterboxPadding>> paddings{this, "paddings"};
    Output<std::vector<NormalizedLandmarkList>> output{this, "output"};

    void process() override;
};

// NORM_RECT + IMAGE_DIMENSIONS mode of LandmarkProjectionCalculator
// (correct for non-square ROIs).
NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const NormalizedRect& rect,
                                        const ImageSize& image_size, bool ignore_rotation = false);

// Projects landmarks from crop coordinates to image coordinates, using either
// the rects the crops were taken from (with `image_size`, exact for any
// aspect ratio; without it, MediaPipe's square-ROI formula) or projection
// matrices.
class LandmarkProjectionNode : public Node {
public:
    LandmarkProjectionNode() : Node("LandmarkProjection") {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<std::vector<NormalizedRect>> rects{this, "rects", /*is_optional=*/true};
    Input<ImageSize> image_size{this, "image_size", /*is_optional=*/true};
    Input<std::vector<Matrix4>> matrices{this, "matrices", /*is_optional=*/true};
    Output<std::vector<NormalizedLandmarkList>> output{this, "output"};

    void process() override;
};

class WorldLandmarkProjectionNode : public Node {
public:
    WorldLandmarkProjectionNode() : Node("WorldLandmarkProjection") {}

    Input<std::vector<LandmarkList>> landmarks{this, "landmarks"};
    Input<std::vector<NormalizedRect>> rects{this, "rects"};
    Output<std::vector<LandmarkList>> output{this, "output"};

    void process() override;
};

// LandmarksToDetectionCalculator: a detection whose box bounds the landmarks
// and whose keypoints are the landmarks.
Detection LandmarksToDetection(const NormalizedLandmarkList& landmarks);

class LandmarksToDetectionNode : public Node {
public:
    LandmarksToDetectionNode() : Node("LandmarksToDetection") {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Output<std::vector<Detection>> detections{this, "detections"};

    void process() override;
};

// LandmarksToTensorCalculator (attributes X, Y; not flattened): one
// [1, num_landmarks, 2] tensor per list, in pixels.
class LandmarksToTensorNode : public Node {
public:
    // `subset` selects landmark indices (all when empty).
    explicit LandmarksToTensorNode(std::vector<int> subset = {})
        : Node("LandmarksToTensor"), subset_(std::move(subset)) {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<ImageSize> image_size{this, "image_size"};
    Output<std::vector<Tensor>> tensors{this, "tensors"};

    void process() override;

private:
    std::vector<int> subset_;
};

}  // namespace mlxpipe
