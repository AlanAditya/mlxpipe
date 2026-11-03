// Smoke and timing test for FaceDetector and FaceLandmarker on synthetic frames.
#include <chrono>
#include <cstdio>
#include <thread>

#include <opencv2/core.hpp>

#include "mlxpipe/tasks/face_detector.h"
#include "mlxpipe/tasks/face_landmarker.h"
#include "test_util.h"

using namespace mlxpipe;

int main() {
    const std::string task = mlxpipe_test::source_dir() + "/tools/models/face_landmarker.task";
    cv::Mat noise(720, 1280, CV_8UC3);
    cv::randu(noise, 0, 255);
    Image frame(noise);

    auto detector = FaceDetector::Create({.model_asset_path = task});
    EXPECT_EQ(detector->Detect(frame).detections.size(), size_t(0));

    auto landmarker = FaceLandmarker::Create({.model_asset_path = task, .output_face_blendshapes = true});
    EXPECT_EQ(landmarker->Detect(frame).face_landmarks.size(), size_t(0));
    double total_ms = 0;
    const int n = 30;
    for (int i = 0; i < n + 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        auto a = std::chrono::steady_clock::now();
        FaceLandmarkerResult r = landmarker->DetectForVideo(frame, 1000 + i * 33);
        auto b = std::chrono::steady_clock::now();
        EXPECT_EQ(r.face_landmarks.size(), size_t(0));
        EXPECT_EQ(landmarker->last_gpu_sync_count(), 1);
        if (i >= 5) total_ms += std::chrono::duration<double, std::milli>(b - a).count();
    }
    std::printf("paced detector-only frame: %.2f ms\n", total_ms / n);
    TEST_MAIN_END();
}
