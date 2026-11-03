#include "mlxpipe/calculators/rect_transformation.h"

#include <algorithm>
#include <cmath>

#include "mlxpipe/calculators/detections_to_rects.h"

namespace mlxpipe {

NormalizedRect TransformNormalizedRect(NormalizedRect rect, const ImageSize& image_size,
                                       const RectTransformationOptions& options) {
    const int image_width = image_size.first;
    const int image_height = image_size.second;
    float width = rect.width;
    float height = rect.height;
    float rotation = rect.rotation;
    if (options.rotation) rotation = NormalizeRadians(rotation + *options.rotation);

    if (rotation == 0.f) {
        rect.x_center += width * options.shift_x;
        rect.y_center += height * options.shift_y;
    } else {
        const float x_shift = (image_width * width * options.shift_x * std::cos(rotation) -
                               image_height * height * options.shift_y * std::sin(rotation)) /
                              image_width;
        const float y_shift = (image_width * width * options.shift_x * std::sin(rotation) +
                               image_height * height * options.shift_y * std::cos(rotation)) /
                              image_height;
        rect.x_center += x_shift;
        rect.y_center += y_shift;
    }

    if (options.square_long) {
        const float long_side = std::max(width * image_width, height * image_height);
        width = long_side / image_width;
        height = long_side / image_height;
    } else if (options.square_short) {
        const float short_side = std::min(width * image_width, height * image_height);
        width = short_side / image_width;
        height = short_side / image_height;
    }
    rect.width = width * options.scale_x;
    rect.height = height * options.scale_y;
    // Like MediaPipe, the adjusted rotation only affects the shift; the
    // rect keeps its original rotation.
    return rect;
}

void RectTransformationNode::process() {
    std::vector<NormalizedRect> out;
    for (const auto& rect : rects.get()) out.push_back(TransformNormalizedRect(rect, image_size.get(), options_));
    output.send(std::move(out));
}

}  // namespace mlxpipe
