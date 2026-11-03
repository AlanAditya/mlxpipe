// Port of mediapipe/calculators/util/detection_projection_calculator.cc.
#pragma once

#include <vector>

#include "mlxpipe/formats/detection.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

// Maps detections through a projection matrix (e.g. ImageToTensor's MATRIX)
// from tensor coordinates to image coordinates.
Detection ProjectDetection(const Detection& detection, const Matrix4& matrix);

class DetectionProjectionNode : public Node {
public:
    DetectionProjectionNode() : Node("DetectionProjection") {}

    Input<std::vector<Detection>> detections{this, "detections"};
    // Uses the first matrix (the detector runs on a single image).
    Input<std::vector<Matrix4>> matrices{this, "matrices"};
    Output<std::vector<Detection>> output{this, "output"};

    void process() override;
};

}  // namespace mlxpipe
