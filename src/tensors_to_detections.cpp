#include "tensors_to_detections.h"
#include <cmath>
#include <stdexcept>

namespace mlx_vision {

std::vector<Detection> DecodeTensorsToDetections(
    const mlx::core::array& box_tensor,
    const mlx::core::array& score_tensor,
    const std::vector<Anchor>& anchors,
    const TensorsToDetectionsOptions& options) 
{
    // Ensure arrays are evaluated and accessible on CPU
    mlx::core::eval(box_tensor);
    mlx::core::eval(score_tensor);
    
    if (anchors.size() != options.num_boxes) {
        throw std::invalid_argument("Number of anchors does not match num_boxes");
    }
    
    const float* raw_boxes = box_tensor.data<float>();
    const float* raw_scores = score_tensor.data<float>();
    
    std::vector<Detection> detections;
    
    for (int i = 0; i < options.num_boxes; ++i) {
        int box_offset = i * options.num_coords + options.box_coord_offset;
        
        float x_center, y_center, w, h;
        if (options.reverse_output_order) {
            y_center = raw_boxes[box_offset];
            x_center = raw_boxes[box_offset + 1];
            h = raw_boxes[box_offset + 2];
            w = raw_boxes[box_offset + 3];
        } else {
            x_center = raw_boxes[box_offset];
            y_center = raw_boxes[box_offset + 1];
            w = raw_boxes[box_offset + 2];
            h = raw_boxes[box_offset + 3];
        }
        
        // Decode coordinates
        x_center = x_center / options.x_scale * anchors[i].w + anchors[i].x_center;
        y_center = y_center / options.y_scale * anchors[i].h + anchors[i].y_center;
        h = h / options.h_scale * anchors[i].h; // MediaPipe uses relative scaling for w/h
        w = w / options.w_scale * anchors[i].w;
        
        // Extract maximum score across classes (assuming single class for now or background class 0)
        float max_score = -std::numeric_limits<float>::max();
        for (int c = 0; c < options.num_classes; ++c) {
            float score = raw_scores[i * options.num_classes + c];
            if (options.sigmoid_score) {
                // Sigmoid is 1 / (1 + exp(-x))
                // Note: MLX can do this in batch, but here we do CPU fallback for demonstration.
                score = 1.0f / (1.0f + std::exp(-score));
            }
            if (score > max_score) {
                max_score = score;
            }
        }
        
        if (max_score >= options.min_score_thresh) {
            Detection det;
            det.x = x_center - w / 2.0f;
            det.y = y_center - h / 2.0f;
            det.width = w;
            det.height = h;
            det.confidence = max_score;
            for (int k = 0; k < options.num_keypoints; ++k) {
                float kx, ky;
                if (options.reverse_output_order) {
                    ky = raw_boxes[box_offset + options.keypoint_coord_offset + k * options.num_values_per_keypoint];
                    kx = raw_boxes[box_offset + options.keypoint_coord_offset + k * options.num_values_per_keypoint + 1];
                } else {
                    kx = raw_boxes[box_offset + options.keypoint_coord_offset + k * options.num_values_per_keypoint];
                    ky = raw_boxes[box_offset + options.keypoint_coord_offset + k * options.num_values_per_keypoint + 1];
                }
                kx = kx / options.x_scale * anchors[i].w + anchors[i].x_center;
                ky = ky / options.y_scale * anchors[i].h + anchors[i].y_center;
                det.keypoints.push_back({kx, ky});
            }
            detections.push_back(det);
        }
    }
    
    return detections;
}

} // namespace mlx_vision
