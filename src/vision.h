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
    float x = 0.0f;
    float y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
    float confidence = 0.0f;
    std::vector<std::pair<float, float>> keypoints;
};

struct Landmark {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct HandResult {
    Detection box; // normalized axis-aligned bounds of the 21 landmarks
    std::vector<Landmark> landmarks; // 21 landmarks
    std::vector<Landmark> world_landmarks; // 21 3D landmarks, metres, hand-centred
    float score = 0.0f;
    // MediaPipe convention: assumes a mirrored (selfie) input image.
    bool is_left_hand = false;
};

// Initialize the MLX backend and load models. Returns true if successful.
MLX_VISION_EXPORT bool initialize_mlx(const std::string& palm_model_path, const std::string& hand_model_path);

// Functional wrapper for hand detection
// Takes an OpenCV mat and returns a list of hand landmarks and bounding boxes
MLX_VISION_EXPORT std::vector<HandResult> detect_hands(const cv::Mat& image);

} // namespace mlx_vision

#endif // MEDIAPIPE_MLX_PORT_VISION_H
