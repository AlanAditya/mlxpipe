// Mirrors mediapipe/framework/formats/detection.proto (relative location data).
#pragma once

#include <vector>

namespace mlxpipe {

struct RelativeBoundingBox {
    float xmin = 0.0f;
    float ymin = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

struct RelativeKeypoint {
    float x = 0.0f;
    float y = 0.0f;
};

struct Detection {
    float score = 0.0f;
    int label_id = 0;
    RelativeBoundingBox box;
    std::vector<RelativeKeypoint> keypoints;
};

}  // namespace mlxpipe
