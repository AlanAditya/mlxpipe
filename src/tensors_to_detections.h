#ifndef MEDIAPIPE_MLX_PORT_TENSORS_TO_DETECTIONS_H
#define MEDIAPIPE_MLX_PORT_TENSORS_TO_DETECTIONS_H

#include <vector>
#include <mlx/mlx.h>
#include "vision.h"
#include "ssd_anchors.h"

namespace mlx_vision {

struct TensorsToDetectionsOptions {
    int num_classes;
    int num_boxes;
    int num_coords;
    float x_scale = 1.0f;
    float y_scale = 1.0f;
    float w_scale = 1.0f;
    float h_scale = 1.0f;
    bool sigmoid_score = true;
    float min_score_thresh = 0.0f;
    int keypoint_coord_offset = 4;
    int num_keypoints = 0;
    int num_values_per_keypoint = 2;
    int box_coord_offset = 0;
    bool reverse_output_order = false;
};

// Decodes the raw model output tensors into bounding boxes using the provided anchors.
// box_tensor must have shape [num_boxes, num_coords]
// score_tensor must have shape [num_boxes, num_classes]
MLX_VISION_EXPORT std::vector<Detection> DecodeTensorsToDetections(
    const mlx::core::array& box_tensor,
    const mlx::core::array& score_tensor,
    const std::vector<Anchor>& anchors,
    const TensorsToDetectionsOptions& options
);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_TENSORS_TO_DETECTIONS_H
