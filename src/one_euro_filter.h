#ifndef ONE_EURO_FILTER_H
#define ONE_EURO_FILTER_H

#include <vector>

class OneEuroFilter {
public:
    OneEuroFilter(double min_cutoff = 1.0, double beta = 0.0, double d_cutoff = 1.0);
    ~OneEuroFilter() = default;

    double filter(double x, double t);
    void reset();

private:
    double alpha(double cutoff, double dt);

    double min_cutoff_;
    double beta_;
    double d_cutoff_;

    bool first_time_;
    double x_prev_;
    double dx_prev_;
    double t_prev_;
};

class LandmarkOneEuroFilter {
public:
    LandmarkOneEuroFilter(int num_landmarks = 21, double min_cutoff = 0.1, double beta = 40.0, double d_cutoff = 1.0);
    ~LandmarkOneEuroFilter() = default;

    void filter(std::vector<float>& xs, std::vector<float>& ys, double t);
    void reset();

private:
    int num_landmarks_;
    std::vector<OneEuroFilter> x_filters_;
    std::vector<OneEuroFilter> y_filters_;
};

#endif // ONE_EURO_FILTER_H
