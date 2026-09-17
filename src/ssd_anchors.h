#ifndef MEDIAPIPE_MLX_PORT_SSD_ANCHORS_H
#define MEDIAPIPE_MLX_PORT_SSD_ANCHORS_H

#include <vector>
#include "vision.h"

namespace mlx_vision {

struct Anchor {
    float x_center;
    float y_center;
    float w;
    float h;
};

struct SsdAnchorsOptions {
    int input_size_width;
    int input_size_height;
    float min_scale;
    float max_scale;
    int num_layers;
    std::vector<int> feature_map_width;
    std::vector<int> feature_map_height;
    std::vector<int> strides;
    std::vector<float> aspect_ratios;
    bool reduce_boxes_in_lowest_layer;
    float interpolated_scale_aspect_ratio;
    bool fixed_anchor_size;
    float anchor_offset_x = 0.5f;
    float anchor_offset_y = 0.5f;
};

// Generates a list of anchors based on the SSD options.
// This replicates MediaPipe's SsdAnchorsCalculator math but purely in C++.
MLX_VISION_EXPORT std::vector<Anchor> GenerateSsdAnchors(const SsdAnchorsOptions& options);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_SSD_ANCHORS_H
