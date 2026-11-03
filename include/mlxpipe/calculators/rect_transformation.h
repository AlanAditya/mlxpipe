// Port of mediapipe/calculators/util/rect_transformation_calculator.cc
// (normalized rects).
#pragma once

#include <optional>
#include <vector>

#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct RectTransformationOptions {
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    // Extra rotation in radians added to the rect's own rotation.
    std::optional<float> rotation;
    float shift_x = 0.0f;
    float shift_y = 0.0f;
    bool square_long = false;
    bool square_short = false;
};

NormalizedRect TransformNormalizedRect(NormalizedRect rect, const ImageSize& image_size,
                                       const RectTransformationOptions& options);

class RectTransformationNode : public Node {
public:
    explicit RectTransformationNode(RectTransformationOptions options)
        : Node("RectTransformation"), options_(options) {}

    Input<std::vector<NormalizedRect>> rects{this, "rects"};
    Input<ImageSize> image_size{this, "image_size"};
    Output<std::vector<NormalizedRect>> output{this, "output"};

    void process() override;

private:
    RectTransformationOptions options_;
};

}  // namespace mlxpipe
