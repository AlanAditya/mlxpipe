// Port of mediapipe/calculators/util/non_max_suppression_calculator.cc.
#pragma once

#include <vector>

#include "mlxpipe/formats/detection.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct NonMaxSuppressionOptions {
    enum class OverlapType { kJaccard, kModifiedJaccard, kIntersectionOverUnion };
    enum class Algorithm { kDefault, kWeighted };

    float min_suppression_threshold = 1.0f;
    OverlapType overlap_type = OverlapType::kJaccard;
    Algorithm algorithm = Algorithm::kDefault;
    int max_num_detections = -1;
    float min_score_threshold = -1.0f;
};

std::vector<Detection> NonMaxSuppression(const std::vector<Detection>& detections,
                                         const NonMaxSuppressionOptions& options);

class NonMaxSuppressionNode : public Node {
public:
    explicit NonMaxSuppressionNode(NonMaxSuppressionOptions options)
        : Node("NonMaxSuppression"), options_(options) {}

    Input<std::vector<Detection>> detections{this, "detections"};
    Output<std::vector<Detection>> output{this, "output"};

    void process() override { output.send(NonMaxSuppression(detections.get(), options_)); }

private:
    NonMaxSuppressionOptions options_;
};

}  // namespace mlxpipe
