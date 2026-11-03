#include "mlxpipe/calculators/hand.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "mlxpipe/calculators/detections_to_rects.h"

namespace mlxpipe {

NormalizedRect HandLandmarksToRect(const NormalizedLandmarkList& all_landmarks, const ImageSize& image_size) {
    // The calculator's joint indices (4, 6, 8) refer to this 12-landmark
    // subset (wrist, thumb base, and each finger's MCP and PIP), so rotation
    // comes from the index/middle/ring knuckles and fingertips don't move the
    // crop. Feeding all 21 landmarks, as the vendored Tasks graph does, makes
    // those indices thumb tip / index PIP / index tip: on recorded webcam
    // frames that lost the second hand in 34 of 123 two-hand frames.
    NormalizedLandmarkList landmarks;
    for (int idx : {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18}) landmarks.push_back(all_landmarks.at(idx));
    constexpr int kWristJoint = 0;
    constexpr int kMiddleFingerPIPJoint = 6;
    constexpr int kIndexFingerPIPJoint = 4;
    constexpr int kRingFingerPIPJoint = 8;
    constexpr float kTargetAngle = M_PI * 0.5f;
    const float w = image_size.first, h = image_size.second;

    const float x0 = landmarks[kWristJoint].x * w;
    const float y0 = landmarks[kWristJoint].y * h;
    float x1 = (landmarks[kIndexFingerPIPJoint].x + landmarks[kRingFingerPIPJoint].x) / 2.f;
    float y1 = (landmarks[kIndexFingerPIPJoint].y + landmarks[kRingFingerPIPJoint].y) / 2.f;
    x1 = (x1 + landmarks[kMiddleFingerPIPJoint].x) / 2.f * w;
    y1 = (y1 + landmarks[kMiddleFingerPIPJoint].y) / 2.f * h;
    const float rotation = NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
    const float reverse_angle = NormalizeRadians(-rotation);

    // MediaPipe initialises the maxima with numeric_limits<float>::min().
    float max_x = std::numeric_limits<float>::min(), max_y = max_x;
    float min_x = std::numeric_limits<float>::max(), min_y = min_x;
    for (const auto& lm : landmarks) {
        max_x = std::max(max_x, lm.x);
        max_y = std::max(max_y, lm.y);
        min_x = std::min(min_x, lm.x);
        min_y = std::min(min_y, lm.y);
    }
    const float axis_aligned_center_x = (max_x + min_x) / 2.f;
    const float axis_aligned_center_y = (max_y + min_y) / 2.f;

    max_x = max_y = std::numeric_limits<float>::min();
    min_x = min_y = std::numeric_limits<float>::max();
    for (const auto& lm : landmarks) {
        const float original_x = (lm.x - axis_aligned_center_x) * w;
        const float original_y = (lm.y - axis_aligned_center_y) * h;
        const float projected_x = original_x * std::cos(reverse_angle) - original_y * std::sin(reverse_angle);
        const float projected_y = original_x * std::sin(reverse_angle) + original_y * std::cos(reverse_angle);
        max_x = std::max(max_x, projected_x);
        max_y = std::max(max_y, projected_y);
        min_x = std::min(min_x, projected_x);
        min_y = std::min(min_y, projected_y);
    }
    const float projected_center_x = (max_x + min_x) / 2.f;
    const float projected_center_y = (max_y + min_y) / 2.f;
    const float center_x = projected_center_x * std::cos(rotation) - projected_center_y * std::sin(rotation) +
                           w * axis_aligned_center_x;
    const float center_y = projected_center_x * std::sin(rotation) + projected_center_y * std::cos(rotation) +
                           h * axis_aligned_center_y;

    NormalizedRect rect;
    rect.x_center = center_x / w;
    rect.y_center = center_y / h;
    rect.width = (max_x - min_x) / w;
    rect.height = (max_y - min_y) / h;
    rect.rotation = rotation;
    return rect;
}

void HandLandmarksToRectNode::process() {
    std::vector<NormalizedRect> out;
    for (size_t i = 0; i < landmarks.get().size(); ++i) {
        out.push_back(HandLandmarksToRect(landmarks.get()[i], image_size.get()));
        if (source_rects.has()) out.back().rect_id = source_rects.get().at(i).rect_id;
    }
    rects.send(std::move(out));
}

float RectIou(const NormalizedRect& a, const NormalizedRect& b) {
    const float ax0 = a.x_center - a.width / 2, ay0 = a.y_center - a.height / 2;
    const float bx0 = b.x_center - b.width / 2, by0 = b.y_center - b.height / 2;
    const float ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    const float ix1 = std::min(ax0 + a.width, bx0 + b.width), iy1 = std::min(ay0 + a.height, by0 + b.height);
    if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
    const float intersection = (ix1 - ix0) * (iy1 - iy0);
    const float normalization = a.width * a.height + b.width * b.height - intersection;
    return normalization > 0.0f ? intersection / normalization : 0.0f;
}

void HandAssociationNode::process() {
    std::vector<NormalizedRect> result;
    std::vector<NormalizedRect> added;
    if (base_rects.has()) {
        for (NormalizedRect rect : base_rects.get()) {
            if (rect.rect_id == 0) rect.rect_id = next_rect_id_++;
            result.push_back(rect);
        }
        if (max_rects_ > 0 && result.size() > max_rects_) result.resize(max_rects_);
    }
    if (rects.has()) {
        for (NormalizedRect rect : rects.get()) {
            if (max_rects_ > 0 && result.size() >= max_rects_) break;
            const bool overlaps = std::any_of(result.begin(), result.end(), [&](const NormalizedRect& existing) {
                return RectIou(existing, rect) > threshold_;
            });
            if (overlaps) continue;
            if (rect.rect_id == 0) rect.rect_id = next_rect_id_++;
            result.push_back(rect);
            added.push_back(rect);
        }
    }
    output.send(std::move(result));
    new_rects.send(std::move(added));
}

namespace {

float Distance(const NormalizedLandmark& a, const NormalizedLandmark& b, int width, int height) {
    return std::sqrt(std::pow((a.x - b.x) * width, 2) + std::pow((a.y - b.y) * height, 2));
}

struct Bound {
    float left, top, right, bottom;
};

Bound CalculateBound(const NormalizedLandmarkList& list) {
    Bound b{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    for (const auto& lm : list) {
        b.left = std::min(b.left, lm.x);
        b.top = std::min(b.top, lm.y);
        b.right = std::max(b.right, lm.x);
        b.bottom = std::max(b.bottom, lm.y);
    }
    return b;
}

// mediapipe/tasks/cc/vision/utils/landmarks_utils.cc CalculateIOU.
float CalculateIou(const Bound& a, const Bound& b) {
    const float area_a = (a.right - a.left) * (a.bottom - a.top);
    const float area_b = (b.right - b.left) * (b.bottom - b.top);
    if (area_a <= 0 || area_b <= 0) return 0.0f;
    const float iw = std::max(0.0f, std::min(a.right, b.right) - std::max(a.left, b.left));
    const float ih = std::max(0.0f, std::min(a.bottom, b.bottom) - std::max(a.top, b.top));
    const float intersection = iw * ih;
    return intersection / (area_a + area_b - intersection);
}

}  // namespace

std::vector<bool> FindDuplicateHands(const std::vector<NormalizedLandmarkList>& hands, const ImageSize& image_size) {
    const int width = image_size.first, height = image_size.second;
    std::vector<float> baselines;
    std::vector<Bound> bounds;
    for (const auto& list : hands) {
        float d = Distance(list[0], list[5], width, height);
        d = std::max(d, Distance(list[5], list[17], width, height));
        d = std::max(d, Distance(list[17], list[0], width, height));
        baselines.push_back(d);
        bounds.push_back(CalculateBound(list));
    }
    std::vector<bool> suppressed(hands.size(), false);
    std::vector<size_t> retained;
    for (size_t i = 0; i < hands.size(); ++i) {
        for (size_t j : retained) {
            const float threshold = std::max(baselines[i], baselines[j]) * 0.2f;
            int matched = 0;
            for (size_t k = 0; k < hands[i].size(); ++k) {
                if (Distance(hands[i][k], hands[j][k], width, height) < threshold) ++matched;
            }
            if (matched >= 10 && CalculateIou(bounds[i], bounds[j]) > 0.2f) {
                suppressed[i] = true;
                break;
            }
        }
        if (!suppressed[i]) retained.push_back(i);
    }
    return suppressed;
}

void HandLandmarksDeduplicationNode::process() {
    std::vector<bool> duplicate = FindDuplicateHands(landmarks.get(), image_size.get());
    std::vector<NormalizedLandmarkList> l;
    std::vector<LandmarkList> w;
    std::vector<NormalizedRect> r;
    std::vector<ClassificationList> c;
    for (size_t i = 0; i < duplicate.size(); ++i) {
        if (duplicate[i]) continue;
        l.push_back(landmarks.get()[i]);
        w.push_back(world_landmarks.get().at(i));
        r.push_back(rois.get().at(i));
        c.push_back(classifications.get().at(i));
    }
    landmarks_out.send(std::move(l));
    world_landmarks_out.send(std::move(w));
    rois_out.send(std::move(r));
    classifications_out.send(std::move(c));
}

}  // namespace mlxpipe
