#include "mlxpipe/calculators/detection_projection.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mlxpipe {

Detection ProjectDetection(const Detection& detection, const Matrix4& m) {
    auto project = [&m](float x, float y) {
        return RelativeKeypoint{x * m[0] + y * m[1] + m[3], x * m[4] + y * m[5] + m[7]};
    };
    Detection out = detection;
    for (auto& kp : out.keypoints) kp = project(kp.x, kp.y);

    const RelativeBoundingBox& b = detection.box;
    const RelativeKeypoint corners[4] = {project(b.xmin, b.ymin), project(b.xmin + b.width, b.ymin),
                                         project(b.xmin + b.width, b.ymin + b.height),
                                         project(b.xmin, b.ymin + b.height)};
    float left = std::numeric_limits<float>::max(), top = left;
    float right = std::numeric_limits<float>::lowest(), bottom = right;
    for (const auto& p : corners) {
        left = std::min(left, p.x);
        top = std::min(top, p.y);
        right = std::max(right, p.x);
        bottom = std::max(bottom, p.y);
    }
    out.box = {left, top, right - left, bottom - top};
    return out;
}

void DetectionProjectionNode::process() {
    if (matrices.get().empty()) throw std::invalid_argument("DetectionProjection: no projection matrix");
    const Matrix4& matrix = matrices.get().front();
    std::vector<Detection> projected;
    for (const auto& detection : detections.get()) projected.push_back(ProjectDetection(detection, matrix));
    output.send(std::move(projected));
}

}  // namespace mlxpipe
