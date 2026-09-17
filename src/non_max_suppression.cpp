#include "non_max_suppression.h"
#include <algorithm>
#include <cmath>

namespace mlx_vision {

static float CalculateOverlapSimilarity(const Detection& rect1, const Detection& rect2) {
    // Both rects defined by (x, y) = top left, (w, h) = dimensions
    // Wait, Detection struct has (x, y, w, h). 
    // Assuming x,y is top-left.
    float r1_xmin = rect1.x;
    float r1_ymin = rect1.y;
    float r1_xmax = rect1.x + rect1.width;
    float r1_ymax = rect1.y + rect1.height;
    
    float r2_xmin = rect2.x;
    float r2_ymin = rect2.y;
    float r2_xmax = rect2.x + rect2.width;
    float r2_ymax = rect2.y + rect2.height;
    
    // Intersection
    float intersect_xmin = std::max(r1_xmin, r2_xmin);
    float intersect_ymin = std::max(r1_ymin, r2_ymin);
    float intersect_xmax = std::min(r1_xmax, r2_xmax);
    float intersect_ymax = std::min(r1_ymax, r2_ymax);
    
    if (intersect_xmax <= intersect_xmin || intersect_ymax <= intersect_ymin) {
        return 0.0f;
    }
    
    float intersection_area = (intersect_xmax - intersect_xmin) * (intersect_ymax - intersect_ymin);
    float area1 = rect1.width * rect1.height;
    float area2 = rect2.width * rect2.height;
    float union_area = area1 + area2 - intersection_area;
    
    if (union_area <= 0.0f) return 0.0f;
    return intersection_area / union_area; // IoU (Jaccard)
}

std::vector<Detection> ApplyNMS(std::vector<Detection>& detections, const NmsOptions& options) {
    std::vector<Detection> output_detections;
    if (detections.empty()) return output_detections;
    
    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        const auto& detection = detections[i];
        if (options.min_score_threshold > 0 && detection.confidence < options.min_score_threshold) {
            break;
        }
        
        output_detections.push_back(detection);
        if (options.max_num_detections > 0 && output_detections.size() >= options.max_num_detections) {
            break;
        }
        
        // Suppress subsequent bounding boxes that overlap too much
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            float similarity = CalculateOverlapSimilarity(detection, detections[j]);
            if (similarity > options.min_suppression_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return output_detections;
}

} // namespace mlx_vision
