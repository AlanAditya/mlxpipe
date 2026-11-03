// Shared helpers for the webcam examples: argument parsing, timing, drawing
// and frame recording.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "mlxpipe/mlxpipe.h"

namespace example {

struct Args {
    std::string models_dir = MLXPIPE_MODELS_DIR;
    int camera = 0;
    int max_items = 2;
    bool mirror = true;
    bool blendshapes = false;
    std::string record_dir;
    int record_every = 5;
    int max_frames = 0;  // 0: until 'q'
};

inline Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--models") args.models_dir = next();
        else if (a == "--camera") args.camera = std::stoi(next());
        else if (a == "--max") args.max_items = std::stoi(next());
        else if (a == "--no-mirror") args.mirror = false;
        else if (a == "--blendshapes") args.blendshapes = true;
        else if (a == "--record") args.record_dir = next();
        else if (a == "--record-every") args.record_every = std::stoi(next());
        else if (a == "--frames") args.max_frames = std::stoi(next());
        else {
            std::fprintf(stderr,
                         "options: --models DIR --camera N --max N --no-mirror --blendshapes "
                         "--record DIR --record-every K --frames N\n");
            std::exit(a == "--help" ? 0 : 1);
        }
    }
    return args;
}

// Strictly increasing milliseconds since the first call.
inline int64_t NowMs() {
    static const auto start = std::chrono::steady_clock::now();
    static int64_t last = -1;
    int64_t ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    last = std::max(ms, last + 1);
    return last;
}

// Rolling timing stats printed every `window` frames.
class Stats {
public:
    explicit Stats(int window = 120) : window_(window) {}
    void Add(double ms) {
        samples_.push_back(ms);
        if (static_cast<int>(samples_.size()) == window_) {
            std::vector<double> s = samples_;
            std::sort(s.begin(), s.end());
            std::printf("[mlxpipe] detect: median %.2f ms, p90 %.2f ms over %d frames\n", s[s.size() / 2],
                        s[s.size() * 9 / 10], window_);
            std::fflush(stdout);
            samples_.clear();
        }
    }

private:
    int window_;
    std::vector<double> samples_;
};

inline cv::Point ToPixel(const mlxpipe::NormalizedLandmark& lm, const cv::Mat& frame) {
    return {static_cast<int>(lm.x * frame.cols), static_cast<int>(lm.y * frame.rows)};
}

inline void DrawLandmarks(cv::Mat& frame, const mlxpipe::NormalizedLandmarkList& landmarks,
                          const std::vector<std::pair<int, int>>& connections, cv::Scalar line_color,
                          cv::Scalar point_color, int point_radius) {
    for (const auto& [a, b] : connections) {
        if (a < static_cast<int>(landmarks.size()) && b < static_cast<int>(landmarks.size())) {
            cv::line(frame, ToPixel(landmarks[a], frame), ToPixel(landmarks[b], frame), line_color, 2, cv::LINE_AA);
        }
    }
    for (const auto& lm : landmarks) cv::circle(frame, ToPixel(lm, frame), point_radius, point_color, -1, cv::LINE_AA);
}

inline void DrawStatus(cv::Mat& frame, double detect_ms, double fps, int gpu_syncs) {
    char text[128];
    std::snprintf(text, sizeof(text), "FPS %.1f | detect %.1f ms | GPU syncs %d", fps, detect_ms, gpu_syncs);
    cv::putText(frame, text, {12, 32}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 0}, 4, cv::LINE_AA);
    cv::putText(frame, text, {12, 32}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2, cv::LINE_AA);
}

// Saves every `every`-th raw frame to `dir` (for offline validation).
class Recorder {
public:
    Recorder(std::string dir, int every) : dir_(std::move(dir)), every_(every) {
        if (!dir_.empty()) std::filesystem::create_directories(dir_);
    }
    void Maybe(const cv::Mat& frame) {
        if (dir_.empty() || frame_++ % every_ != 0) return;
        char name[64];
        std::snprintf(name, sizeof(name), "/frame_%05d.png", saved_++);
        cv::imwrite(dir_ + name, frame);
    }

private:
    std::string dir_;
    int every_;
    int frame_ = 0;
    int saved_ = 0;
};

// Runs a capture loop, calling `process(frame, timestamp_ms)` which detects
// and draws, and returns the GPU sync count.
template <typename Process>
int RunCamera(const Args& args, const char* window, Process process) {
    cv::VideoCapture capture(args.camera);
    if (!capture.isOpened()) {
        std::fprintf(stderr, "Could not open camera %d (check camera permissions for your terminal)\n", args.camera);
        return 1;
    }
    Recorder recorder(args.record_dir, args.record_every);
    Stats stats;
    double fps = 0;
    auto last = std::chrono::steady_clock::now();
    cv::Mat frame;
    for (int n = 0; args.max_frames == 0 || n < args.max_frames; ++n) {
        if (!capture.read(frame) || frame.empty()) break;
        if (args.mirror) cv::flip(frame, frame, 1);
        recorder.Maybe(frame);

        auto t0 = std::chrono::steady_clock::now();
        int syncs = process(frame, NowMs());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        stats.Add(ms);

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        fps = fps == 0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;
        DrawStatus(frame, ms, fps, syncs);
        cv::imshow(window, frame);
        if (cv::waitKey(1) == 'q') break;
    }
    return 0;
}

}  // namespace example
