#ifndef MEDIAPIPE_MLX_PORT_VISION_H
#define MEDIAPIPE_MLX_PORT_VISION_H

#include <opencv2/core.hpp>
#include <vector>

namespace mlx_vision {

// Export macro for dynamic library
#if defined(_WIN32)
#define MLX_VISION_EXPORT __declspec(dllexport)
#else
#define MLX_VISION_EXPORT __attribute__((visibility("default")))
#endif

// A simple structure to represent a detection
struct Detection {
    float x;
    float y;
    float width;
    float height;
    float confidence;
    std::vector<std::pair<float, float>> keypoints;
};

struct Landmark {
    float x;
    float y;
    float z;
};

struct HandResult {
    Detection box;
    std::vector<Landmark> landmarks; // 21 landmarks
    std::vector<Landmark> world_landmarks; // 21 3D landmarks
    float score;
    bool is_left_hand;
};

// Initialize the MLX backend and load models. Returns true if successful.
MLX_VISION_EXPORT bool initialize_mlx(const std::string& palm_model_path, const std::string& hand_model_path);

// Functional wrapper for hand detection
// Takes an OpenCV mat and returns a list of hand landmarks and bounding boxes
MLX_VISION_EXPORT std::vector<HandResult> detect_hands(const cv::Mat& image);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_VISION_H
