#include "one_euro_filter.h"
#include <cmath>

OneEuroFilter::OneEuroFilter(double min_cutoff, double beta, double d_cutoff)
    : min_cutoff_(min_cutoff), beta_(beta), d_cutoff_(d_cutoff), first_time_(true), x_prev_(0.0), dx_prev_(0.0), t_prev_(0.0) {}

void OneEuroFilter::reset() {
    first_time_ = true;
    x_prev_ = 0.0;
    dx_prev_ = 0.0;
    t_prev_ = 0.0;
}

double OneEuroFilter::alpha(double cutoff, double dt) {
    double tau = 1.0 / (2.0 * M_PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
}

double OneEuroFilter::filter(double x, double t) {
    if (first_time_) {
        first_time_ = false;
        x_prev_ = x;
        dx_prev_ = 0.0;
        t_prev_ = t;
        return x;
    }

    double dt = t - t_prev_;
    if (dt <= 0.0) {
        dt = 1e-5; // Prevent division by zero
    }

    // Estimate the velocity (derivative)
    double dx = (x - x_prev_) / dt;

    // Filter the velocity
    double alpha_d = alpha(d_cutoff_, dt);
    double dx_hat = alpha_d * dx + (1.0 - alpha_d) * dx_prev_;

    // Calculate cutoff frequency based on velocity
    double cutoff = min_cutoff_ + beta_ * std::abs(dx_hat);

    // Filter the data
    double alpha_x = alpha(cutoff, dt);
    double x_hat = alpha_x * x + (1.0 - alpha_x) * x_prev_;

    // Update state
    x_prev_ = x_hat;
    dx_prev_ = dx_hat;
    t_prev_ = t;

    return x_hat;
}


LandmarkOneEuroFilter::LandmarkOneEuroFilter(int num_landmarks, double min_cutoff, double beta, double d_cutoff)
    : num_landmarks_(num_landmarks) {
    for (int i = 0; i < num_landmarks_; ++i) {
        x_filters_.emplace_back(min_cutoff, beta, d_cutoff);
        y_filters_.emplace_back(min_cutoff, beta, d_cutoff);
    }
}

void LandmarkOneEuroFilter::reset() {
    for (int i = 0; i < num_landmarks_; ++i) {
        x_filters_[i].reset();
        y_filters_[i].reset();
    }
}

void LandmarkOneEuroFilter::filter(std::vector<float>& xs, std::vector<float>& ys, double t) {
    if (xs.size() != num_landmarks_ || ys.size() != num_landmarks_) return;

    for (int i = 0; i < num_landmarks_; ++i) {
        xs[i] = static_cast<float>(x_filters_[i].filter(static_cast<double>(xs[i]), t));
        ys[i] = static_cast<float>(y_filters_[i].filter(static_cast<double>(ys[i]), t));
    }
}
