// Mirrors mediapipe/framework/formats/classification.proto.
#pragma once

#include <string>
#include <vector>

namespace mlxpipe {

struct Classification {
    int index = 0;
    float score = 0.0f;
    std::string label;
    std::string display_name;
};
using ClassificationList = std::vector<Classification>;

}  // namespace mlxpipe
