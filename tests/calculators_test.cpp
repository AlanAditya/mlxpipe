// Unit tests for calculator math against MediaPipe's conventions.
#include <cmath>

#include "mlxpipe/calculators/detections_to_rects.h"
#include "mlxpipe/calculators/hand.h"
#include "mlxpipe/calculators/image_to_tensor.h"
#include "mlxpipe/calculators/landmarks.h"
#include "mlxpipe/calculators/non_max_suppression.h"
#include "mlxpipe/calculators/rect_transformation.h"
#include "test_util.h"

using namespace mlxpipe;

// An upright synthetic hand: wrist at the bottom, fingers pointing up.
static NormalizedLandmarkList UprightHand(float cx, float cy, float s) {
    NormalizedLandmarkList hand(21);
    hand[0] = {cx, cy + 0.5f * s};
    for (int f = 0; f < 5; ++f) {
        float x = cx + (f - 2) * 0.12f * s;
        for (int j = 0; j < 4; ++j) hand[1 + f * 4 + j] = {x, cy + 0.2f * s - j * 0.15f * s};
    }
    return hand;
}

int main() {
    const ImageSize size{1920, 1080};

    // HandLandmarksToRect: an upright hand gives ~0 rotation, and moving the
    // fingertips (thumb tip 4, index tip 8) must not rotate the crop.
    {
        NormalizedLandmarkList hand = UprightHand(0.5f, 0.5f, 0.3f);
        NormalizedRect rect = HandLandmarksToRect(hand, size);
        EXPECT_NEAR(rect.rotation, 0.0, 0.05);
        NormalizedLandmarkList moved = hand;
        moved[4].x += 0.1f;
        moved[8].x -= 0.1f;
        moved[8].y += 0.05f;
        EXPECT_NEAR(HandLandmarksToRect(moved, size).rotation, rect.rotation, 1e-6);
    }

    // DetectionsToRects rotation: keypoint vector pointing up with a 90 degree
    // target gives 0; pointing right gives +90 degrees (clockwise).
    {
        DetectionsToRectsOptions options;
        options.rotation_vector_start_keypoint_index = 0;
        options.rotation_vector_end_keypoint_index = 1;
        options.rotation_vector_target_angle = M_PI / 2;
        Detection up;
        up.box = {0.4f, 0.4f, 0.2f, 0.2f};
        up.keypoints = {{0.5f, 0.6f}, {0.5f, 0.4f}};
        EXPECT_NEAR(DetectionToNormalizedRect(up, size, options).rotation, 0.0, 1e-6);
        Detection right = up;
        right.keypoints = {{0.4f, 0.5f}, {0.6f, 0.5f}};
        EXPECT_NEAR(DetectionToNormalizedRect(right, size, options).rotation, M_PI / 2, 1e-6);
    }

    // RectTransformation: square_long in pixels, then scale; shift moves the
    // centre along the rect's own (rotated) y axis.
    {
        NormalizedRect rect{0.5f, 0.5f, 100.0f / 1920, 200.0f / 1080, 0.0f};
        RectTransformationOptions options;
        options.square_long = true;
        options.scale_x = options.scale_y = 2.0f;
        options.shift_y = -0.5f;
        NormalizedRect out = TransformNormalizedRect(rect, size, options);
        EXPECT_NEAR(out.width * 1920, 400.0, 1e-3);
        EXPECT_NEAR(out.height * 1080, 400.0, 1e-3);
        EXPECT_NEAR(out.y_center * 1080, 540.0 - 100.0, 1e-3);
        rect.rotation = M_PI / 2;
        out = TransformNormalizedRect(rect, size, options);
        EXPECT_NEAR(out.x_center * 1920, 960.0 + 100.0, 1e-3);
        EXPECT_NEAR(out.rotation, M_PI / 2, 1e-6);  // rotation is not re-written
    }

    // ImageToTensor's matrix and LandmarkProjection agree for a rotated,
    // non-square ROI.
    {
        NormalizedRect rect{0.3f, 0.6f, 0.2f, 0.25f, 0.7f};
        RotatedRect roi = GetRoi(size.first, size.second, rect);
        Matrix4 m = GetRotatedSubRectToRectTransformMatrix(roi, size.first, size.second);
        NormalizedLandmarkList crop = {{0.1f, 0.2f, 0.05f}, {0.9f, 0.7f, -0.02f}, {0.5f, 0.5f, 0.0f}};
        auto a = ProjectLandmarks(crop, m);
        auto b = ProjectLandmarks(crop, rect, size);
        for (size_t i = 0; i < crop.size(); ++i) {
            EXPECT_NEAR(a[i].x, b[i].x, 1e-5);
            EXPECT_NEAR(a[i].y, b[i].y, 1e-5);
        }
        EXPECT_NEAR(a[2].x, rect.x_center, 1e-5);  // crop centre maps to rect centre
        EXPECT_NEAR(a[2].y, rect.y_center, 1e-5);
    }

    // PadRoi letterboxes a wide region into a square tensor.
    {
        RotatedRect roi{960, 540, 1920, 1080, 0};
        LetterboxPadding p = PadRoi(192, 192, true, &roi);
        EXPECT_NEAR(roi.width, 1920.0, 1e-3);
        EXPECT_NEAR(roi.height, 1920.0, 1e-3);
        EXPECT_NEAR(p.top, (1.0 - 1080.0 / 1920.0) / 2, 1e-6);
        EXPECT_NEAR(p.left, 0.0, 1e-6);
    }

    // Weighted NMS blends overlapping boxes by score; plain NMS keeps the best.
    {
        Detection a, b, c;
        a.score = 0.9f;
        a.box = {0.10f, 0.10f, 0.20f, 0.20f};
        b.score = 0.6f;
        b.box = {0.12f, 0.12f, 0.20f, 0.20f};
        c.score = 0.8f;
        c.box = {0.70f, 0.70f, 0.10f, 0.10f};
        NonMaxSuppressionOptions options;
        options.min_suppression_threshold = 0.3f;
        options.overlap_type = NonMaxSuppressionOptions::OverlapType::kIntersectionOverUnion;
        auto plain = NonMaxSuppression({a, b, c}, options);
        EXPECT_EQ(plain.size(), size_t(2));
        EXPECT_NEAR(plain[0].box.xmin, 0.10, 1e-6);
        options.algorithm = NonMaxSuppressionOptions::Algorithm::kWeighted;
        auto weighted = NonMaxSuppression({a, b, c}, options);
        EXPECT_EQ(weighted.size(), size_t(2));
        EXPECT_NEAR(weighted[0].box.xmin, (0.10 * 0.9 + 0.12 * 0.6) / 1.5, 1e-6);
        EXPECT_NEAR(weighted[0].score, 0.9, 1e-6);
    }

    // Hand deduplication: an identical hand is a duplicate, a distant one not.
    {
        NormalizedLandmarkList h1 = UprightHand(0.3f, 0.5f, 0.3f);
        NormalizedLandmarkList h2 = UprightHand(0.7f, 0.5f, 0.3f);
        auto dup = FindDuplicateHands({h1, h2, h1}, size);
        EXPECT_TRUE(!dup[0] && !dup[1] && dup[2]);
    }
    TEST_MAIN_END();
}
