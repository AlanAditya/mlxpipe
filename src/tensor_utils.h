#ifndef MEDIAPIPE_MLX_PORT_TENSOR_UTILS_H
#define MEDIAPIPE_MLX_PORT_TENSOR_UTILS_H

#include <opencv2/core.hpp>
#include <mlx/mlx.h>
#include "vision.h"

namespace mlx_vision {

// Converts a cv::Mat (expected BGR uint8) to an MLX array (RGB float32)
// normalizes pixel values to [0, 1] or [-1, 1] depending on the model's requirement.
// By default, scales [0, 255] to [0.0, 1.0].
MLX_VISION_EXPORT mlx::core::array cvMat_to_mlxArray(const cv::Mat& image);

// Converts an MLX array (RGB float32) back to a cv::Mat (BGR uint8).
// Assumes the MLX array values are in [0.0, 1.0].
MLX_VISION_EXPORT cv::Mat mlxArray_to_cvMat(const mlx::core::array& tensor);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_TENSOR_UTILS_H
