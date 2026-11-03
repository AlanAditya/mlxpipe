// Port of mediapipe/calculators/tensor/tensors_to_detections_calculator.cc
// (CPU path, single-image batch).
#pragma once

#include <vector>

#include "mlxpipe/calculators/ssd_anchors.h"
#include "mlxpipe/formats/detection.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct TensorsToDetectionsOptions {
    enum class BoxFormat { kUnspecified, kYXHW, kXYWH, kXYXY };

    int num_classes = 1;
    int num_boxes = 0;
    int num_coords = 0;
    int keypoint_coord_offset = 0;
    int num_keypoints = 0;
    int num_values_per_keypoint = 2;
    int box_coord_offset = 0;
    float x_scale = 0.0f;
    float y_scale = 0.0f;
    float w_scale = 0.0f;
    float h_scale = 0.0f;
    bool apply_exponential_on_box_size = false;
    // Without box_format, true selects XYWH and false YXHW.
    bool reverse_output_order = false;
    BoxFormat box_format = BoxFormat::kUnspecified;
    bool sigmoid_score = false;
    float score_clipping_thresh = 0.0f;  // 0 disables clipping
    bool flip_vertically = false;
    float min_score_thresh = 0.0f;
    int max_results = -1;
};

// Decodes raw boxes [num_boxes * num_coords] and scores [num_boxes * num_classes].
std::vector<Detection> DecodeDetections(const float* raw_boxes, const float* raw_scores,
                                        const std::vector<Anchor>& anchors, const TensorsToDetectionsOptions& options);

class TensorsToDetectionsNode : public Node {
public:
    // `box_tensor_index` / `score_tensor_index` select the model outputs.
    TensorsToDetectionsNode(TensorsToDetectionsOptions options, std::vector<Anchor> anchors, int box_tensor_index = 0,
                            int score_tensor_index = 1);

    Input<std::vector<Tensor>> tensors{this, "tensors"};
    Output<std::vector<Detection>> detections{this, "detections"};

    bool reads_tensors() const override { return true; }
    void process() override;

private:
    TensorsToDetectionsOptions options_;
    std::vector<Anchor> anchors_;
    int box_index_;
    int score_index_;
};

}  // namespace mlxpipe
