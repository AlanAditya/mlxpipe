// Port of mediapipe/calculators/tensor/image_to_tensor_calculator.cc.
//
// Crops (optionally rotated) regions of an image into a float tensor. Unlike
// MediaPipe, which converts one region per invocation inside a loop, the
// node crops every rect into a single [N, H, W, 3] batch.
#pragma once

#include <array>
#include <optional>
#include <vector>

#include "mlxpipe/formats/image.h"
#include "mlxpipe/formats/rect.h"
#include "mlxpipe/framework/graph.h"

namespace mlxpipe {

struct ImageToTensorOptions {
    enum class BorderMode { kZero, kReplicate };

    int output_tensor_width = 0;
    int output_tensor_height = 0;
    // Letterbox the region to the tensor's aspect ratio instead of stretching.
    bool keep_aspect_ratio = false;
    float output_tensor_float_range_min = 0.0f;
    float output_tensor_float_range_max = 1.0f;
    // MediaPipe's default is BORDER_REPLICATE.
    BorderMode border_mode = BorderMode::kReplicate;
};

// A region in pixels, as in image_to_tensor_utils.h.
struct RotatedRect {
    float center_x = 0.0f;
    float center_y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f;
};

// GetRoi: the pixel region of `rect`, or the whole image without one.
RotatedRect GetRoi(int image_width, int image_height, const std::optional<NormalizedRect>& rect);
// PadRoi: grows `roi` to the tensor aspect ratio; returns the letterbox padding.
LetterboxPadding PadRoi(int tensor_width, int tensor_height, bool keep_aspect_ratio, RotatedRect* roi);
// GetRotatedSubRectToRectTransformMatrix: maps normalized tensor coordinates
// to normalized image coordinates.
Matrix4 GetRotatedSubRectToRectTransformMatrix(const RotatedRect& sub_rect, int rect_width, int rect_height,
                                               bool flip_horizontally = false);

// Crops `roi` out of `image` into `dst` (uint8 RGB, tensor size), using the
// same corner mapping as MediaPipe's OpenCV converter.
void CropRoi(const Image& image, const RotatedRect& roi, const ImageToTensorOptions& options, cv::Mat& dst);

class ImageToTensorNode : public Node {
public:
    explicit ImageToTensorNode(ImageToTensorOptions options);

    Input<Image> image{this, "image"};
    // Regions to crop. When unconnected the whole image is used (one item).
    Input<std::vector<NormalizedRect>> rects{this, "rects", /*is_optional=*/true};

    // [N, H, W, 3] float32 in the configured range. Not produced when `rects`
    // is connected but empty.
    Output<Tensor> tensor{this, "tensor"};
    Output<std::vector<Matrix4>> matrices{this, "matrices"};
    Output<std::vector<LetterboxPadding>> letterbox_paddings{this, "letterbox_paddings"};
    Output<ImageSize> image_size{this, "image_size"};

    void process() override;

private:
    ImageToTensorOptions options_;
};

// ImagePropertiesCalculator: the image size.
class ImagePropertiesNode : public Node {
public:
    ImagePropertiesNode() : Node("ImageProperties") {}

    Input<Image> image{this, "image"};
    Output<ImageSize> size{this, "size"};

    void process() override { size.send(image.get().size()); }
};

}  // namespace mlxpipe
