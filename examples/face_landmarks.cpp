// Webcam face mesh (478 landmarks) with mlxpipe::FaceLandmarker.
// --blendshapes also shows the strongest blendshape scores.
#include <algorithm>

#include "example_util.h"

int main(int argc, char** argv) {
    example::Args args = example::ParseArgs(argc, argv);
    auto landmarker = mlxpipe::FaceLandmarker::Create({.model_asset_path = args.models_dir + "/face_landmarker.task",
                                                       .num_faces = args.max_items,
                                                       .output_face_blendshapes = args.blendshapes});
    const auto contours = mlxpipe::FaceContourConnections();

    return example::RunCamera(args, "mlxpipe face landmarks", [&](cv::Mat& frame, int64_t timestamp_ms) {
        auto result = landmarker->DetectForVideo(mlxpipe::Image(frame), timestamp_ms);
        for (size_t i = 0; i < result.face_landmarks.size(); ++i) {
            for (const auto& lm : result.face_landmarks[i]) {
                cv::circle(frame, example::ToPixel(lm, frame), 1, {200, 200, 200}, -1);
            }
            example::DrawLandmarks(frame, result.face_landmarks[i], contours, {0, 220, 255}, {0, 220, 255}, 0);
            if (i < result.face_blendshapes.size()) {
                auto scores = result.face_blendshapes[i];
                std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
                for (int k = 0; k < 6 && k < static_cast<int>(scores.size()); ++k) {
                    char text[64];
                    std::snprintf(text, sizeof(text), "%-20s %.2f", scores[k].label.c_str(), scores[k].score);
                    cv::putText(frame, text, {12, 70 + 28 * k}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2,
                                cv::LINE_AA);
                }
            }
        }
        return landmarker->last_gpu_sync_count();
    });
}
