// Mirrors mediapipe/framework/formats/rect.proto.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace mlxpipe {

// A rotated rectangle in coordinates normalized to [0, 1] by image width and
// height. `rotation` is clockwise in radians around the centre.
struct NormalizedRect {
    float x_center = 0.0f;
    float y_center = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float rotation = 0.0f;
    // Optional identifier; 0 means unset.
    int64_t rect_id = 0;
};

// (width, height) in pixels, as in MediaPipe's IMAGE_SIZE streams.
using ImageSize = std::pair<int, int>;

// Row-major 4x4 matrix, as produced by ImageToTensorCalculator's MATRIX output.
using Matrix4 = std::vector<float>;

// Letterbox padding normalized to the tensor: left, top, right, bottom.
struct LetterboxPadding {
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

}  // namespace mlxpipe
