#include "mlxpipe/calculators/detections_to_rects.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlxpipe {

float NormalizeRadians(float angle) { return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI)); }

NormalizedRect DetectionToNormalizedRect(const Detection& detection, const ImageSize& image_size,
                                         const DetectionsToRectsOptions& options) {
    NormalizedRect rect;
    if (options.conversion_mode == DetectionsToRectsOptions::ConversionMode::kUseKeypoints) {
        float xmin = std::numeric_limits<float>::max(), ymin = xmin;
        float xmax = std::numeric_limits<float>::lowest(), ymax = xmax;
        for (const auto& kp : detection.keypoints) {
            xmin = std::min(xmin, kp.x);
            ymin = std::min(ymin, kp.y);
            xmax = std::max(xmax, kp.x);
            ymax = std::max(ymax, kp.y);
        }
        rect.x_center = (xmin + xmax) / 2;
        rect.y_center = (ymin + ymax) / 2;
        rect.width = xmax - xmin;
        rect.height = ymax - ymin;
    } else {
        rect.x_center = detection.box.xmin + detection.box.width / 2;
        rect.y_center = detection.box.ymin + detection.box.height / 2;
        rect.width = detection.box.width;
        rect.height = detection.box.height;
    }
    if (options.rotation_vector_start_keypoint_index && options.rotation_vector_end_keypoint_index) {
        const auto& start = detection.keypoints.at(*options.rotation_vector_start_keypoint_index);
        const auto& end = detection.keypoints.at(*options.rotation_vector_end_keypoint_index);
        const float x0 = start.x * image_size.first;
        const float y0 = start.y * image_size.second;
        const float x1 = end.x * image_size.first;
        const float y1 = end.y * image_size.second;
        rect.rotation = NormalizeRadians(options.rotation_vector_target_angle - std::atan2(-(y1 - y0), x1 - x0));
    }
    return rect;
}

void DetectionsToRectsNode::process() {
    std::vector<NormalizedRect> out;
    for (size_t i = 0; i < detections.get().size(); ++i) {
        out.push_back(DetectionToNormalizedRect(detections.get()[i], image_size.get(), options_));
        if (source_rects.has()) out.back().rect_id = source_rects.get().at(i).rect_id;
    }
    rects.send(std::move(out));
}

}  // namespace mlxpipe
