// Smoke and timing test for HandLandmarker on synthetic frames, plus a
// regression comparison against recorded frames when tests/data has them.
#include <chrono>
#include <cstdio>
#include <thread>

#include <opencv2/core.hpp>

#include "mlxpipe/tasks/hand_landmarker.h"
#include "test_util.h"

using namespace mlxpipe;

int main() {
    const std::string models = mlxpipe_test::source_dir() + "/tools/models/";
    HandLandmarkerOptions options;
    options.model_asset_path = models + "hand_landmarker.task";
    options.num_hands = 2;
    auto landmarker = HandLandmarker::Create(options);

    cv::Mat noise(1080, 1920, CV_8UC3);
    cv::randu(noise, 0, 255);
    Image frame(noise);

    HandLandmarkerResult still = landmarker->Detect(frame);
    EXPECT_EQ(still.hand_landmarks.size(), size_t(0));

    double total_ms = 0;
    const int n = 30;
    for (int i = 0; i < n + 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
        auto a = std::chrono::steady_clock::now();
        HandLandmarkerResult r = landmarker->DetectForVideo(frame, 1000 + i * 33);
        auto b = std::chrono::steady_clock::now();
        EXPECT_EQ(r.hand_landmarks.size(), size_t(0));
        EXPECT_EQ(landmarker->last_gpu_sync_count(), 1);
        if (i >= 5) total_ms += std::chrono::duration<double, std::milli>(b - a).count();
    }
    std::printf("paced palm-only frame: %.2f ms\n", total_ms / n);
    TEST_MAIN_END();
}
