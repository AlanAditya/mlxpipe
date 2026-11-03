// Port of mediapipe/calculators/tflite/ssd_anchors_calculator.cc.
#pragma once

#include <vector>

namespace mlxpipe {

struct Anchor {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
};

struct SsdAnchorsOptions {
    int input_size_width = 0;
    int input_size_height = 0;
    float min_scale = 0.0f;
    float max_scale = 0.0f;
    float anchor_offset_x = 0.5f;
    float anchor_offset_y = 0.5f;
    int num_layers = 0;
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;
    std::vector<int> strides;
    std::vector<float> aspect_ratios;
    bool reduce_boxes_in_lowest_layer = false;
    float interpolated_scale_aspect_ratio = 1.0f;
    bool fixed_anchor_size = false;
};

std::vector<Anchor> GenerateSsdAnchors(const SsdAnchorsOptions& options);

}  // namespace mlxpipe
