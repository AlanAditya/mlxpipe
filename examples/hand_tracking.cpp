// Webcam hand tracking with mlxpipe::HandLandmarker.
#include "example_util.h"

int main(int argc, char** argv) {
    example::Args args = example::ParseArgs(argc, argv);
    auto landmarker = mlxpipe::HandLandmarker::Create(
        {.model_asset_path = args.models_dir + "/hand_landmarker.task", .num_hands = args.max_items});

    return example::RunCamera(args, "mlxpipe hand tracking", [&](cv::Mat& frame, int64_t timestamp_ms) {
        auto result = landmarker->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
        for (size_t i = 0; i < result.hand_landmarks.size(); ++i) {
            const auto& landmarks = result.hand_landmarks[i];
            example::DrawLandmarks(frame, landmarks, mlxpipe::kHandConnections, {255, 160, 0}, {0, 0, 255}, 4);
            const auto& top = result.handedness[i].front();
            char label[48];
            std::snprintf(label, sizeof(label), "%s %.2f", top.label.c_str(), top.score);
            cv::putText(frame, label, example::ToPixel(landmarks[0], frame) + cv::Point(-30, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2, cv::LINE_AA);
        }
        return landmarker->last_gpu_sync_count();
    });
}
