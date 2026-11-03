#include "mlxpipe/calculators/landmarks_smoothing.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace mlxpipe {

float LowPassFilter::Apply(float value) {
    float result;
    if (initialized_) {
        result = alpha_ * value + (1.0 - alpha_) * stored_value_;
    } else {
        result = value;
        initialized_ = true;
    }
    raw_value_ = value;
    stored_value_ = result;
    return result;
}

float LowPassFilter::ApplyWithAlpha(float value, float alpha) {
    SetAlpha(alpha);
    return Apply(value);
}

void LowPassFilter::SetAlpha(float alpha) {
    if (alpha < 0.0f || alpha > 1.0f) return;  // MediaPipe logs and keeps the old alpha
    alpha_ = alpha;
}

OneEuroFilter::OneEuroFilter(double frequency, double min_cutoff, double beta, double derivate_cutoff)
    : frequency_(frequency),
      min_cutoff_(min_cutoff),
      beta_(beta),
      derivate_cutoff_(derivate_cutoff),
      x_(GetAlpha(min_cutoff)),
      dx_(GetAlpha(derivate_cutoff)) {}

double OneEuroFilter::GetAlpha(double cutoff) const {
    double te = 1.0 / frequency_;
    double tau = 1.0 / (2 * M_PI * cutoff);
    return 1.0 / (1.0 + tau / te);
}

double OneEuroFilter::Apply(int64_t timestamp_ns, double value, double value_scale, double beta_scale) {
    if (last_time_ >= timestamp_ns) return value;
    if (last_time_ != 0 && timestamp_ns != 0) frequency_ = 1.0 / ((timestamp_ns - last_time_) * 1e-9);
    last_time_ = timestamp_ns;
    double dvalue = x_.HasLastRawValue() ? (value - x_.LastRawValue()) * value_scale * frequency_ : 0.0;
    double edvalue = dx_.ApplyWithAlpha(dvalue, GetAlpha(derivate_cutoff_));
    double cutoff = min_cutoff_ + beta_scale * beta_ * std::fabs(edvalue);
    return x_.ApplyWithAlpha(value, GetAlpha(cutoff));
}

NormalizedLandmarkList LandmarksOneEuroSmoother::Apply(const NormalizedLandmarkList& landmarks,
                                                       const ImageSize& image_size, int64_t timestamp_us) {
    const float w = image_size.first, h = image_size.second;
    if (x_.empty()) {
        for (size_t i = 0; i < landmarks.size(); ++i) {
            x_.emplace_back(options_.frequency, options_.min_cutoff, options_.beta, options_.derivate_cutoff);
            y_.emplace_back(options_.frequency, options_.min_cutoff, options_.beta, options_.derivate_cutoff);
            z_.emplace_back(options_.frequency, options_.min_cutoff, options_.beta, options_.derivate_cutoff);
        }
    }
    // Filter in pixels (z scaled like x).
    float value_scale = 1.0f;
    if (!options_.disable_value_scaling) {
        float x_min = w, x_max = 0, y_min = h, y_max = 0;
        x_min = y_min = std::numeric_limits<float>::max();
        x_max = y_max = std::numeric_limits<float>::lowest();
        for (const auto& lm : landmarks) {
            x_min = std::min(x_min, lm.x * w);
            x_max = std::max(x_max, lm.x * w);
            y_min = std::min(y_min, lm.y * h);
            y_max = std::max(y_max, lm.y * h);
        }
        const float object_scale = ((x_max - x_min) + (y_max - y_min)) / 2.0f;
        if (object_scale < options_.min_allowed_object_scale) return landmarks;
        value_scale = 1.0f / object_scale;
    }
    const int64_t timestamp_ns = timestamp_us * 1000;
    NormalizedLandmarkList out = landmarks;
    for (size_t i = 0; i < landmarks.size(); ++i) {
        out[i].x = x_[i].Apply(timestamp_ns, landmarks[i].x * w, value_scale) / w;
        out[i].y = y_[i].Apply(timestamp_ns, landmarks[i].y * h, value_scale) / h;
        out[i].z = z_[i].Apply(timestamp_ns, landmarks[i].z * w, value_scale) / w;
    }
    return out;
}

void LandmarksSmoothingNode::process() {
    std::map<int64_t, LandmarksOneEuroSmoother> active;
    std::vector<NormalizedLandmarkList> out;
    for (size_t i = 0; i < landmarks.get().size(); ++i) {
        const int64_t id = rects.get().at(i).rect_id;
        auto it = smoothers_.find(id);
        LandmarksOneEuroSmoother smoother = it != smoothers_.end() ? std::move(it->second)
                                                                  : LandmarksOneEuroSmoother(options_);
        out.push_back(smoother.Apply(landmarks.get()[i], image_size.get(), timestamp()));
        active.emplace(id, std::move(smoother));
    }
    smoothers_ = std::move(active);
    output.send(std::move(out));
}

}  // namespace mlxpipe
