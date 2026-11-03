// An image frame. Wraps a cv::Mat without copying.
#pragma once

#include <opencv2/core.hpp>

#include "mlxpipe/formats/rect.h"

namespace mlxpipe {

enum class ImageFormat {
    kSRGB,   // 8-bit RGB
    kSBGR,   // 8-bit BGR, OpenCV's default
    kSRGBA,  // 8-bit RGBA
    kSBGRA,  // 8-bit BGRA
};

struct Image {
    Image() = default;
    // `mat` must be 8-bit with 3 or 4 channels matching `format`.
    Image(cv::Mat image_mat, ImageFormat image_format = ImageFormat::kSBGR)
        : mat(std::move(image_mat)), format(image_format) {}

    int width() const { return mat.cols; }
    int height() const { return mat.rows; }
    ImageSize size() const { return {mat.cols, mat.rows}; }

    cv::Mat mat;
    ImageFormat format = ImageFormat::kSBGR;
};

}  // namespace mlxpipe
