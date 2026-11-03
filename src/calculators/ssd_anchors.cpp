#include "mlxpipe/calculators/ssd_anchors.h"

#include <cmath>

namespace mlxpipe {

namespace {

float CalculateScale(float min_scale, float max_scale, int stride_index, int num_strides) {
    if (num_strides == 1) return (min_scale + max_scale) * 0.5f;
    return min_scale + (max_scale - min_scale) * 1.0 * stride_index / (num_strides - 1.0f);
}

}  // namespace

std::vector<Anchor> GenerateSsdAnchors(const SsdAnchorsOptions& options) {
    std::vector<Anchor> anchors;
    const int num_strides = options.strides.size();
    int layer_id = 0;
    while (layer_id < options.num_layers) {
        std::vector<float> anchor_height;
        std::vector<float> anchor_width;
        std::vector<float> aspect_ratios;
        std::vector<float> scales;

        // For same strides, merge the anchors in the same order.
        int last_same_stride_layer = layer_id;
        while (last_same_stride_layer < num_strides &&
               options.strides[last_same_stride_layer] == options.strides[layer_id]) {
            const float scale = CalculateScale(options.min_scale, options.max_scale, last_same_stride_layer, num_strides);
            if (last_same_stride_layer == 0 && options.reduce_boxes_in_lowest_layer) {
                aspect_ratios.insert(aspect_ratios.end(), {1.0f, 2.0f, 0.5f});
                scales.insert(scales.end(), {0.1f, scale, scale});
            } else {
                for (float ratio : options.aspect_ratios) {
                    aspect_ratios.push_back(ratio);
                    scales.push_back(scale);
                }
                if (options.interpolated_scale_aspect_ratio > 0.0f) {
                    const float scale_next =
                        last_same_stride_layer == num_strides - 1
                            ? 1.0f
                            : CalculateScale(options.min_scale, options.max_scale, last_same_stride_layer + 1,
                                             num_strides);
                    scales.push_back(std::sqrt(scale * scale_next));
                    aspect_ratios.push_back(options.interpolated_scale_aspect_ratio);
                }
            }
            last_same_stride_layer++;
        }

        for (size_t i = 0; i < aspect_ratios.size(); ++i) {
            const float ratio_sqrts = std::sqrt(aspect_ratios[i]);
            anchor_height.push_back(scales[i] / ratio_sqrts);
            anchor_width.push_back(scales[i] * ratio_sqrts);
        }

        int feature_map_height = 0;
        int feature_map_width = 0;
        if (!options.feature_map_height.empty()) {
            feature_map_height = options.feature_map_height[layer_id];
            feature_map_width = options.feature_map_width[layer_id];
        } else {
            const int stride = options.strides[layer_id];
            feature_map_height = std::ceil(1.0f * options.input_size_height / stride);
            feature_map_width = std::ceil(1.0f * options.input_size_width / stride);
        }

        for (int y = 0; y < feature_map_height; ++y) {
            for (int x = 0; x < feature_map_width; ++x) {
                for (size_t anchor_id = 0; anchor_id < anchor_height.size(); ++anchor_id) {
                    Anchor anchor;
                    anchor.x_center = (x + options.anchor_offset_x) * 1.0f / feature_map_width;
                    anchor.y_center = (y + options.anchor_offset_y) * 1.0f / feature_map_height;
                    if (options.fixed_anchor_size) {
                        anchor.w = 1.0f;
                        anchor.h = 1.0f;
                    } else {
                        anchor.w = anchor_width[anchor_id];
                        anchor.h = anchor_height[anchor_id];
                    }
                    anchors.push_back(anchor);
                }
            }
        }
        layer_id = last_same_stride_layer;
    }
    return anchors;
}

}  // namespace mlxpipe
