#include "mlxpipe/calculators/image_to_tensor.h"

#include <cmath>
#include <opencv2/geometry/2d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace mlxpipe {

namespace mx = mlx::core;

RotatedRect GetRoi(int image_width, int image_height, const std::optional<NormalizedRect>& rect) {
    if (rect) {
        return {rect->x_center * image_width, rect->y_center * image_height, rect->width * image_width,
                rect->height * image_height, rect->rotation};
    }
    return {0.5f * image_width, 0.5f * image_height, static_cast<float>(image_width),
            static_cast<float>(image_height), 0.0f};
}

LetterboxPadding PadRoi(int tensor_width, int tensor_height, bool keep_aspect_ratio, RotatedRect* roi) {
    if (!keep_aspect_ratio) return {};
    const float tensor_aspect_ratio = static_cast<float>(tensor_height) / tensor_width;
    const float roi_aspect_ratio = roi->height / roi->width;
    float vertical_padding = 0.0f;
    float horizontal_padding = 0.0f;
    float new_width;
    float new_height;
    if (tensor_aspect_ratio > roi_aspect_ratio) {
        new_width = roi->width;
        new_height = roi->width * tensor_aspect_ratio;
        vertical_padding = (1.0f - roi_aspect_ratio / tensor_aspect_ratio) / 2.0f;
    } else {
        new_width = roi->height / tensor_aspect_ratio;
        new_height = roi->height;
        horizontal_padding = (1.0f - tensor_aspect_ratio / roi_aspect_ratio) / 2.0f;
    }
    roi->width = new_width;
    roi->height = new_height;
    return {horizontal_padding, vertical_padding, horizontal_padding, vertical_padding};
}

Matrix4 GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect, int rect_width, int rect_height,
                                               bool flip_horizontally) {
    Matrix4 matrix(16, 0.0f);
    const float a = sub_rect.width;
    const float b = sub_rect.height;
    const float flip = flip_horizontally ? -1 : 1;
    const float c = std::cos(sub_rect.rotation);
    const float d = std::sin(sub_rect.rotation);
    const float e = sub_rect.center_x;
    const float f = sub_rect.center_y;
    const float g = 1.0f / rect_width;
    const float h = 1.0f / rect_height;
    matrix[0] = a * c * flip * g;
    matrix[1] = -b * d * g;
    matrix[3] = (-0.5f * a * c * flip + 0.5f * b * d + e) * g;
    matrix[4] = a * d * flip * h;
    matrix[5] = b * c * h;
    matrix[7] = (-0.5f * b * c - 0.5f * a * d * flip + f) * h;
    matrix[10] = a * g;
    matrix[15] = 1.0f;
    return matrix;
}

void CropRoi(const Image& image, const RotatedRect& roi, const ImageToTensorOptions& options, cv::Mat& dst) {
    if (std::isnan(roi.center_x) || std::isnan(roi.center_y) || std::isnan(roi.width) || std::isnan(roi.height) ||
        std::isnan(roi.rotation) || roi.width <= 0 || roi.height <= 0) {
        throw std::invalid_argument("ImageToTensor: invalid ROI");
    }
    // Same correspondence as ImageToTensorOpenCvConverter: the rotated rect's
    // box points map to the tensor's corners (0,H), (0,0), (W,0).
    const cv::RotatedRect rotated_rect(cv::Point2f(roi.center_x, roi.center_y), cv::Size2f(roi.width, roi.height),
                                       roi.rotation * 180.f / M_PI);
    cv::Point2f src[4];
    rotated_rect.points(src);
    const float w = options.output_tensor_width;
    const float h = options.output_tensor_height;
    const cv::Point2f dst_points[3] = {{0.0f, h}, {0.0f, 0.0f}, {w, 0.0f}};
    cv::Mat transform = cv::getAffineTransform(src, dst_points);

    int border = options.border_mode == ImageToTensorOptions::BorderMode::kZero ? cv::BORDER_CONSTANT
                                                                                : cv::BORDER_REPLICATE;
    cv::Mat warped;
    cv::warpAffine(image.mat, warped, transform, cv::Size(options.output_tensor_width, options.output_tensor_height),
                   cv::INTER_LINEAR, border, cv::Scalar(0, 0, 0, 0));
    switch (image.format) {
        case ImageFormat::kSRGB: warped.copyTo(dst); break;
        case ImageFormat::kSBGR: cv::cvtColor(warped, dst, cv::COLOR_BGR2RGB); break;
        case ImageFormat::kSRGBA: cv::cvtColor(warped, dst, cv::COLOR_RGBA2RGB); break;
        case ImageFormat::kSBGRA: cv::cvtColor(warped, dst, cv::COLOR_BGRA2RGB); break;
    }
}

ImageToTensorNode::ImageToTensorNode(ImageToTensorOptions options)
    : Node("ImageToTensor"), options_(std::move(options)) {
    if (options_.output_tensor_width <= 0 || options_.output_tensor_height <= 0) {
        throw std::invalid_argument("ImageToTensor: output tensor size must be positive");
    }
}

void ImageToTensorNode::process() {
    const Image& img = image.get();
    const int w = options_.output_tensor_width;
    const int h = options_.output_tensor_height;

    std::vector<std::optional<NormalizedRect>> regions;
    if (rects.connected()) {
        if (!rects.has()) return;
        for (const auto& r : rects.get()) regions.emplace_back(r);
    } else {
        regions.emplace_back(std::nullopt);
    }
    image_size.send(img.size());
    if (regions.empty()) {
        matrices.send({});
        letterbox_paddings.send({});
        return;
    }

    const int n = regions.size();
    cv::Mat batch(n * h, w, CV_8UC3);
    std::vector<Matrix4> mats;
    std::vector<LetterboxPadding> paddings;
    std::vector<RotatedRect> rois;
    for (int i = 0; i < n; ++i) {
        RotatedRect roi = GetRoi(img.width(), img.height(), regions[i]);
        paddings.push_back(PadRoi(w, h, options_.keep_aspect_ratio, &roi));
        mats.push_back(GetRotatedSubRectToRectTransformMatrix(roi, img.width(), img.height()));
        rois.push_back(roi);
    }
    // The crops are small: OpenCV's thread pool costs more than it saves (its workers spin after
    // every warpAffine, doubling the process's CPU time at 30 fps). A single-stripe parallel_for_
    // runs inline, and OpenCV runs parallel loops nested in it serially, without touching the
    // process-wide cv::setNumThreads.
    cv::parallel_for_(cv::Range(0, 1), [&](const cv::Range&) {
        for (int i = 0; i < n; ++i) {
            cv::Mat dst = batch.rowRange(i * h, (i + 1) * h);
            CropRoi(img, rois[i], options_, dst);
        }
    }, 1);

    // Upload uint8 and map [0, 255] to the output range on the GPU.
    const float scale = (options_.output_tensor_float_range_max - options_.output_tensor_float_range_min) / 255.0f;
    mx::array pixels(batch.data, {n, h, w, 3}, mx::uint8);
    tensor.send(mx::add(mx::multiply(mx::astype(pixels, mx::float32), mx::array(scale)),
                        mx::array(options_.output_tensor_float_range_min)));
    matrices.send(std::move(mats));
    letterbox_paddings.send(std::move(paddings));
}

}  // namespace mlxpipe
