// Port of mediapipe/calculators/util/landmarks_smoothing_calculator.cc
// (one-euro filter) and mediapipe/util/filtering/{low_pass,one_euro}_filter.cc.
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "mlxpipe/formats/landmark.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

class LowPassFilter {
public:
    explicit LowPassFilter(float alpha) { SetAlpha(alpha); }
    float Apply(float value);
    float ApplyWithAlpha(float value, float alpha);
    bool HasLastRawValue() const { return initialized_; }
    float LastRawValue() const { return raw_value_; }

private:
    void SetAlpha(float alpha);
    float alpha_ = 1.0f;
    float raw_value_ = 0.0f;
    float stored_value_ = 0.0f;
    bool initialized_ = false;
};

class OneEuroFilter {
public:
    OneEuroFilter(double frequency, double min_cutoff, double beta, double derivate_cutoff);
    // `timestamp_ns` must increase between calls.
    double Apply(int64_t timestamp_ns, double value, double value_scale, double beta_scale = 1.0);

private:
    double GetAlpha(double cutoff) const;
    double frequency_;
    double min_cutoff_;
    double beta_;
    double derivate_cutoff_;
    LowPassFilter x_;
    LowPassFilter dx_;
    int64_t last_time_ = -1;
};

struct OneEuroFilterOptions {
    float frequency = 30.0f;
    float min_cutoff = 1.0f;
    float beta = 0.0f;
    float derivate_cutoff = 1.0f;
    float min_allowed_object_scale = 1e-6f;
    bool disable_value_scaling = false;
};

// Smooths one landmark list in pixel space, scaling velocities by the
// object's size (OneEuroFilterImpl in landmarks_smoothing_calculator_utils.cc).
class LandmarksOneEuroSmoother {
public:
    explicit LandmarksOneEuroSmoother(OneEuroFilterOptions options) : options_(options) {}
    NormalizedLandmarkList Apply(const NormalizedLandmarkList& landmarks, const ImageSize& image_size,
                                 int64_t timestamp_us);

private:
    OneEuroFilterOptions options_;
    std::vector<OneEuroFilter> x_, y_, z_;
};

// Smooths each tracked object's landmarks with its own filter, keyed by the
// rect_id of `rects`. MediaPipe only smooths a single face; per-id filters
// extend that to several. Filters of ids that disappear are dropped.
class LandmarksSmoothingNode : public Node {
public:
    explicit LandmarksSmoothingNode(OneEuroFilterOptions options)
        : Node("LandmarksSmoothing"), options_(options) {}

    Input<std::vector<NormalizedLandmarkList>> landmarks{this, "landmarks"};
    Input<std::vector<NormalizedRect>> rects{this, "rects"};
    Input<ImageSize> image_size{this, "image_size"};
    Output<std::vector<NormalizedLandmarkList>> output{this, "output"};

    void process() override;

private:
    OneEuroFilterOptions options_;
    std::map<int64_t, LandmarksOneEuroSmoother> smoothers_;
};

}  // namespace mlxpipe
