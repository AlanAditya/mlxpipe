// Hand-specific calculators:
//   HandLandmarksToRect        mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc
//   HandAssociation            mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_association_calculator.cc
//   HandLandmarksDeduplication mediapipe/tasks/cc/vision/hand_landmarker/calculators/hand_landmarks_deduplication_calculator.cc
#pragma once

#include <vector>

#include "mlxpipe/formats/classification.h"
#include "mlxpipe/formats/landmark.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

// Hand ROI from 21 landmarks, using the knuckle subset the calculator was
// designed for (see hand.cpp).
NormalizedRect HandLandmarksToRect(const NormalizedLandmarkList& landmarks, const ImageSize& image_size);

class HandLandmarksToRectNode : public Node {
public:
    HandLandmarksToRectNode() : Node("HandLandmarksToRect") {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<ImageSize> image_size{this, "image_size"};
    // Optional: the rects the landmarks came from; their rect_id is carried
    // over so a hand keeps its id while tracked.
    Input<std::vector<NormalizedRect>> source_rects{this, "source_rects", /*is_optional=*/true};
    Output<std::vector<NormalizedRect>> rects{this, "rects"};

    void process() override;
};

// Axis-aligned IoU of two rects, ignoring rotation (mediapipe/util/rectangle_util.cc).
float RectIou(const NormalizedRect& a, const NormalizedRect& b);

// Keeps every base rect, then appends each new rect whose IoU with all kept
// rects is at most `min_similarity_threshold`. Rects without an id get one.
class HandAssociationNode : public Node {
public:
    // `max_rects` > 0 clips the result like the ClipNormalizedRectVectorSize
    // node that follows association in MediaPipe.
    explicit HandAssociationNode(float min_similarity_threshold, size_t max_rects = 0)
        : Node("HandAssociation"), threshold_(min_similarity_threshold), max_rects_(max_rects) {}

    Input<std::vector<NormalizedRect>> base_rects{this, "base_rects", /*is_optional=*/true};
    Input<std::vector<NormalizedRect>> rects{this, "rects", /*is_optional=*/true};
    Output<std::vector<NormalizedRect>> output{this, "output"};
    // Only the rects added from `rects` (the batched landmark pass needs
    // just the new ones; base rects were already processed).
    Output<std::vector<NormalizedRect>> new_rects{this, "new_rects"};

    void process() override;

private:
    float threshold_;
    size_t max_rects_;
    int64_t next_rect_id_ = 1;
};

// Indices of hands that duplicate an earlier hand: at least 10 of 21
// landmarks within 20% of the larger palm size and landmark-box IoU > 0.2.
std::vector<bool> FindDuplicateHands(const std::vector<NormalizedLandmarkList>& hands, const ImageSize& image_size);

class HandLandmarksDeduplicationNode : public Node {
public:
    HandLandmarksDeduplicationNode() : Node("HandLandmarksDeduplication") {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<std::vector<LandmarkList>> world_landmarks{this, "world_landmarks"};
    Input<std::vector<NormalizedRect>> rois{this, "rois"};
    Input<std::vector<ClassificationList>> classifications{this, "classifications"};
    Input<ImageSize> image_size{this, "image_size"};

    Output<std::vector<NormalizedLandmarkList>> landmarks_out{this, "landmarks_out"};
    Output<std::vector<LandmarkList>> world_landmarks_out{this, "world_landmarks_out"};
    Output<std::vector<NormalizedRect>> rois_out{this, "rois_out"};
    Output<std::vector<ClassificationList>> classifications_out{this, "classifications_out"};

    void process() override;
};

}  // namespace mlxpipe
