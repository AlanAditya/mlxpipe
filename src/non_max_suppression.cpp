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

// Port of MediaPipe's NonMaxSuppressionCalculator::WeightedNonMaxSuppression.
// Expects detections sorted by confidence, descending.
static std::vector<Detection> WeightedNMS(const std::vector<Detection>& detections, const NmsOptions& options) {
    std::vector<Detection> output_detections;
    std::vector<size_t> remained(detections.size());
    for (size_t i = 0; i < remained.size(); ++i) remained[i] = i;

    while (!remained.empty()) {
        const Detection& detection = detections[remained[0]];
        if (options.min_score_threshold > 0 && detection.confidence < options.min_score_threshold) {
            break;
        }

        std::vector<size_t> candidates, rest;
        for (size_t idx : remained) {
            if (CalculateOverlapSimilarity(detections[idx], detection) > options.min_suppression_threshold) {
                candidates.push_back(idx);
            } else {
                rest.push_back(idx);
            }
        }

        Detection weighted = detection;
        if (!candidates.empty()) {
            float w_xmin = 0, w_ymin = 0, w_xmax = 0, w_ymax = 0, total_score = 0;
            std::vector<std::pair<float, float>> keypoints(detection.keypoints.size(), {0.0f, 0.0f});
            for (size_t idx : candidates) {
                const Detection& c = detections[idx];
                total_score += c.confidence;
                w_xmin += c.x * c.confidence;
                w_ymin += c.y * c.confidence;
                w_xmax += (c.x + c.width) * c.confidence;
                w_ymax += (c.y + c.height) * c.confidence;
                for (size_t k = 0; k < keypoints.size(); ++k) {
                    keypoints[k].first += c.keypoints[k].first * c.confidence;
                    keypoints[k].second += c.keypoints[k].second * c.confidence;
                }
            }
            weighted.x = w_xmin / total_score;
            weighted.y = w_ymin / total_score;
            weighted.width = w_xmax / total_score - weighted.x;
            weighted.height = w_ymax / total_score - weighted.y;
            for (size_t k = 0; k < keypoints.size(); ++k) {
                weighted.keypoints[k] = {keypoints[k].first / total_score, keypoints[k].second / total_score};
            }
        }

        output_detections.push_back(weighted);
        if (options.max_num_detections > 0 && output_detections.size() >= options.max_num_detections) {
            break;
        }
        if (rest.size() == remained.size()) break;
        remained = std::move(rest);
    }
    return output_detections;
}

std::vector<Detection> ApplyNMS(std::vector<Detection>& detections, const NmsOptions& options) {
    std::vector<Detection> output_detections;
    if (detections.empty()) return output_detections;
    
    // Sort by confidence descending
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b) {
        return a.confidence > b.confidence;
    });
    
    if (options.weighted) return WeightedNMS(detections, options);
    
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
