// Replays a folder of frames (e.g. recorded with an example's --record) as a
// video through HandLandmarker and FaceLandmarker.
//
//   mlxpipe_replay <frames_dir> [--overlays DIR] [--save FILE] [--compare FILE]
//                               [--save-faces FILE] [--compare-faces FILE]
//
// --save / --save-faces write the hand / face results; --compare /
// --compare-faces report how a run differs from a saved one (counts, mean
// landmark distance in pixels), which catches tracking regressions on real
// footage. The files are plain text (per frame: count, then x y for every
// landmark), so results from other implementations can be compared too.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "mlxpipe/mlxpipe.h"

using namespace mlxpipe;
namespace fs = std::filesystem;

using FrameLists = std::vector<NormalizedLandmarkList>;

static void Save(const std::string& path, const std::vector<FrameLists>& frames) {
    std::ofstream out(path);
    for (const auto& lists : frames) {
        out << lists.size();
        for (const auto& list : lists) {
            for (const auto& lm : list) out << ' ' << lm.x << ' ' << lm.y;
        }
        out << '\n';
    }
}

// The landmark count per list is implied by the line length.
static std::vector<FrameLists> Load(const std::string& path) {
    std::ifstream in(path);
    std::vector<FrameLists> frames;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream s(line);
        size_t n = 0;
        s >> n;
        std::vector<float> values;
        for (float v; s >> v;) values.push_back(v);
        FrameLists lists(n, NormalizedLandmarkList(n ? values.size() / 2 / n : 0));
        size_t i = 0;
        for (auto& list : lists) {
            for (auto& lm : list) lm.x = values[i++], lm.y = values[i++];
        }
        frames.push_back(lists);
    }
    return frames;
}

// Pairs each saved list with the closest run list by landmark `anchor` (within
// 100 px) and averages the per-landmark distance over the pairs.
static void Compare(const char* what, const std::string& path, const std::vector<FrameLists>& results, int anchor,
                    int width, int height) {
    std::vector<FrameLists> base = Load(path);
    auto dist = [&](const NormalizedLandmark& a, const NormalizedLandmark& b) {
        return std::hypot((a.x - b.x) * width, (a.y - b.y) * height);
    };
    int count_match = 0, matched = 0;
    double total = 0;
    std::vector<double> per_list;
    for (size_t f = 0; f < std::min(base.size(), results.size()); ++f) {
        count_match += base[f].size() == results[f].size();
        for (const auto& b : base[f]) {
            const NormalizedLandmarkList* best = nullptr;
            double best_d = 1e9;
            for (const auto& r : results[f]) {
                if (r.size() != b.size()) continue;
                double d = dist(b[anchor], r[anchor]);
                if (d < best_d) best_d = d, best = &r;
            }
            if (!best || best_d > 100) continue;
            double e = 0;
            for (size_t k = 0; k < b.size(); ++k) e += dist(b[k], (*best)[k]);
            per_list.push_back(e / b.size());
            total += e / b.size();
            ++matched;
        }
    }
    std::sort(per_list.begin(), per_list.end());
    auto pct = [&](double p) { return per_list.empty() ? 0.0 : per_list[size_t(p * (per_list.size() - 1))]; };
    std::printf("%s vs %s: count agrees %d/%zu | mean landmark distance %.2f px over %d %s (median %.2f, p95 %.2f)\n",
                what, path.c_str(), count_match, base.size(), matched ? total / matched : 0.0, matched, what, pct(0.5),
                pct(0.95));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: mlxpipe_replay <frames_dir> [--overlays DIR] [--save FILE] [--compare FILE] [--save-faces FILE] [--compare-faces FILE]\n");
        return 1;
    }
    std::string overlays, save, compare, save_faces, compare_faces;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--overlays") overlays = argv[i + 1];
        else if (a == "--save") save = argv[i + 1];
        else if (a == "--compare") compare = argv[i + 1];
        else if (a == "--save-faces") save_faces = argv[i + 1];
        else if (a == "--compare-faces") compare_faces = argv[i + 1];
    }
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(argv[1])) {
        if (e.path().extension() == ".png" || e.path().extension() == ".jpg") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (!overlays.empty()) fs::create_directories(overlays);

    const std::string models = MLXPIPE_MODELS_DIR;
    auto hands = HandLandmarker::Create({.model_asset_path = models + "/hand_landmarker.task", .num_hands = 2});
    auto faces = FaceLandmarker::Create({.model_asset_path = models + "/face_landmarker.task"});
    const auto contours = FaceContourConnections();

    std::vector<FrameLists> hand_results, face_results;
    std::vector<double> hand_ms, face_ms;
    int frames_with_hands = 0, frames_with_faces = 0;
    int64_t timestamp_ms = 0;
    for (const auto& file : files) {
        cv::Mat frame = cv::imread(file.string());
        timestamp_ms += 33;
        auto t0 = std::chrono::steady_clock::now();
        HandLandmarkerResult h = hands->DetectForVideo(Image(frame), timestamp_ms);
        auto t1 = std::chrono::steady_clock::now();
        FaceLandmarkerResult f = faces->DetectForVideo(Image(frame), timestamp_ms);
        auto t2 = std::chrono::steady_clock::now();
        hand_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        face_ms.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        frames_with_hands += !h.hand_landmarks.empty();
        frames_with_faces += !f.face_landmarks.empty();
        hand_results.push_back(h.hand_landmarks);
        face_results.push_back(f.face_landmarks);

        if (overlays.empty()) continue;
        auto px = [&](const NormalizedLandmark& lm) { return cv::Point(lm.x * frame.cols, lm.y * frame.rows); };
        for (const auto& list : f.face_landmarks) {
            for (const auto& [a, b] : contours) cv::line(frame, px(list[a]), px(list[b]), {0, 220, 255}, 2);
        }
        for (size_t i = 0; i < h.hand_landmarks.size(); ++i) {
            const auto& list = h.hand_landmarks[i];
            for (const auto& [a, b] : kHandConnections) cv::line(frame, px(list[a]), px(list[b]), {0, 255, 0}, 2);
            cv::putText(frame, h.handedness[i][0].label, px(list[0]) + cv::Point(0, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        {0, 255, 0}, 2);
        }
        cv::imwrite(overlays + "/" + file.filename().string(), frame);
    }

    auto median = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };
    std::printf("frames %zu | with hands %d | with faces %d | median ms: hands %.2f, face %.2f\n", files.size(),
                frames_with_hands, frames_with_faces, median(hand_ms), median(face_ms));
    if (!save.empty()) Save(save, hand_results);
    if (!save_faces.empty()) Save(save_faces, face_results);
    if (files.empty()) return 0;
    cv::Mat first = cv::imread(files.front().string());
    // Anchors: the wrist for hands, the nose tip for faces.
    if (!compare.empty()) Compare("hands", compare, hand_results, 0, first.cols, first.rows);
    if (!compare_faces.empty()) Compare("faces", compare_faces, face_results, 1, first.cols, first.rows);
    return 0;
}
