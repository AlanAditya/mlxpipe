#ifndef MEDIAPIPE_MLX_PORT_NON_MAX_SUPPRESSION_H
#define MEDIAPIPE_MLX_PORT_NON_MAX_SUPPRESSION_H

#include <vector>
#include "vision.h"

namespace mlx_vision {

struct NmsOptions {
    float min_suppression_threshold = 0.3f;
    int max_num_detections = -1;
    float min_score_threshold = 0.0f;
};

// Applies Non-Maximum Suppression to filter out overlapping bounding boxes.
// Detections should already be decoded into absolute or relative coords (x,y,w,h).
MLX_VISION_EXPORT std::vector<Detection> ApplyNMS(std::vector<Detection>& detections, const NmsOptions& options);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_NON_MAX_SUPPRESSION_H
