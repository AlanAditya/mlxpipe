// Port of mediapipe/calculators/util/detections_to_rects_calculator.cc
// (normalized output).
#pragma once

#include <optional>
#include <vector>

#include "mlxpipe/formats/detection.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct DetectionsToRectsOptions {
    enum class ConversionMode { kUseBoundingBox, kUseKeypoints };

    // When set, the rect is rotated so the vector from the start keypoint to
    // the end keypoint points at `rotation_vector_target_angle` (radians).
    std::optional<int> rotation_vector_start_keypoint_index;
    std::optional<int> rotation_vector_end_keypoint_index;
    float rotation_vector_target_angle = 0.0f;
    ConversionMode conversion_mode = ConversionMode::kUseBoundingBox;
};

float NormalizeRadians(float angle);
NormalizedRect DetectionToNormalizedRect(const Detection& detection, const ImageSize& image_size,
                                         const DetectionsToRectsOptions& options);

class DetectionsToRectsNode : public Node {
public:
    explicit DetectionsToRectsNode(DetectionsToRectsOptions options)
        : Node("DetectionsToRects"), options_(std::move(options)) {}

    Input<std::vector<Detection>> detections{this, "detections"};
    Input<ImageSize> image_size{this, "image_size"};
    // Optional: copies rect_id from these rects (one per detection).
    Input<std::vector<NormalizedRect>> source_rects{this, "source_rects", /*is_optional=*/true};
    Output<std::vector<NormalizedRect>> rects{this, "rects"};

    void process() override;

private:
    DetectionsToRectsOptions options_;
};

}  // namespace mlxpipe
