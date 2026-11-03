// Port of mediapipe/calculators/util/association_norm_rect_calculator.cc.
#pragma once

#include <vector>

#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

// Merges `base_rects` (e.g. tracked from the previous frame) with `rects`
// (e.g. fresh detections). Rects are added in order; an added rect removes
// every earlier rect it overlaps (IoU > min_similarity_threshold) and inherits
// the removed rect's id. Rects without an id get a new one.
class AssociationNormRectNode : public Node {
public:
    explicit AssociationNormRectNode(float min_similarity_threshold, size_t max_rects = 0)
        : Node("AssociationNormRect"), threshold_(min_similarity_threshold), max_rects_(max_rects) {}

    Input<std::vector<NormalizedRect>> base_rects{this, "base_rects", /*is_optional=*/true};
    Input<std::vector<NormalizedRect>> rects{this, "rects", /*is_optional=*/true};

    // The associated rects, clipped to `max_rects` when set.
    Output<std::vector<NormalizedRect>> output{this, "output"};
    // Which base rects survive in `output`, in base order.
    Output<std::vector<bool>> base_kept{this, "base_kept"};
    // The rects in `output` that came from `rects`.
    Output<std::vector<NormalizedRect>> new_rects{this, "new_rects"};

    void process() override;

private:
    float threshold_;
    size_t max_rects_;
    int64_t next_rect_id_ = 1;
};

}  // namespace mlxpipe
