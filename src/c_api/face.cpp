// The C API in include/mlxpipe_face.h, over FaceDetector and FaceLandmarker.
#include "mlxpipe_face.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlxpipe/tasks/connections.h"
#include "mlxpipe/tasks/face_detector.h"
#include "mlxpipe/tasks/face_landmarker.h"

struct mlxpipe_face_detector {
    std::unique_ptr<mlxpipe::FaceDetector> impl;
};

struct mlxpipe_face_landmarker {
    std::unique_ptr<mlxpipe::FaceLandmarker> impl;
};

namespace {

thread_local std::string g_last_error;

// Runs `f`, turning exceptions into `on_error` and a message for
// mlxpipe_last_error().
template <typename F, typename R>
R Guard(F&& f, R on_error) {
    try {
        g_last_error.clear();
        return f();
    } catch (const std::exception& e) {
        g_last_error = e.what();
    } catch (...) {
        g_last_error = "unknown error";
    }
    return on_error;
}

// Wraps the caller's pixels without copying.
mlxpipe::Image ToImage(const mlxpipe_image* image) {
    if (!image || !image->data || image->width <= 0 || image->height <= 0) {
        throw std::invalid_argument("mlxpipe: invalid image");
    }
    mlxpipe::ImageFormat format;
    int channels;
    switch (image->format) {
        case MLXPIPE_PIXEL_RGB: format = mlxpipe::ImageFormat::kSRGB, channels = 3; break;
        case MLXPIPE_PIXEL_BGR: format = mlxpipe::ImageFormat::kSBGR, channels = 3; break;
        case MLXPIPE_PIXEL_RGBA: format = mlxpipe::ImageFormat::kSRGBA, channels = 4; break;
        case MLXPIPE_PIXEL_BGRA: format = mlxpipe::ImageFormat::kSBGRA, channels = 4; break;
        default: throw std::invalid_argument("mlxpipe: unknown pixel format");
    }
    const size_t stride = image->bytes_per_row > 0 ? image->bytes_per_row : size_t(image->width) * channels;
    if (stride < size_t(image->width) * channels) throw std::invalid_argument("mlxpipe: bytes_per_row too small");
    cv::Mat mat(image->height, image->width, CV_8UC(channels), const_cast<uint8_t*>(image->data), stride);
    return mlxpipe::Image(mat, format);
}

void Require(const void* p, const char* what) {
    if (!p) throw std::invalid_argument(std::string("mlxpipe: ") + what + " is NULL");
}

int CopyLandmarks(const mlxpipe::FaceLandmarkerResult& result, mlxpipe_face_landmarks* out, int capacity) {
    const int n = result.face_landmarks.size();
    if (n > 0 && capacity > 0) Require(out, "out");
    for (int i = 0; i < std::min(n, capacity); ++i) {
        const auto& list = result.face_landmarks[i];
        float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
        for (int k = 0; k < MLXPIPE_FACE_NUM_LANDMARKS && k < int(list.size()); ++k) {
            out[i].landmarks[k] = {list[k].x, list[k].y, list[k].z};
            x0 = std::min(x0, list[k].x), x1 = std::max(x1, list[k].x);
            y0 = std::min(y0, list[k].y), y1 = std::max(y1, list[k].y);
        }
        out[i].xmin = x0, out[i].ymin = y0, out[i].width = x1 - x0, out[i].height = y1 - y0;
        std::fill(std::begin(out[i].blendshapes), std::end(out[i].blendshapes), 0.0f);
        if (i < int(result.face_blendshapes.size())) {
            for (const auto& c : result.face_blendshapes[i]) {
                if (c.index >= 0 && c.index < MLXPIPE_FACE_NUM_BLENDSHAPES) out[i].blendshapes[c.index] = c.score;
            }
        }
    }
    return n;
}

// Each region's landmarks in contour order, each listed once.
std::vector<int> RegionIndices(const std::vector<std::pair<int, int>>& connections) {
    std::vector<int> out;
    for (const auto& [a, b] : connections) {
        for (int v : {a, b}) {
            if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
        }
    }
    return out;
}

}  // namespace

extern "C" {

const char* mlxpipe_last_error(void) { return g_last_error.c_str(); }

void mlxpipe_face_detector_default_options(mlxpipe_face_detector_options* options) {
    if (!options) return;
    mlxpipe::FaceDetectorOptions d;
    *options = {nullptr, d.min_detection_confidence, d.min_suppression_threshold, d.num_faces};
}

mlxpipe_face_detector* mlxpipe_face_detector_create(const mlxpipe_face_detector_options* options) {
    return Guard(
        [&]() -> mlxpipe_face_detector* {
            Require(options, "options");
            Require(options->model_path, "options->model_path");
            mlxpipe::FaceDetectorOptions o;
            o.model_asset_path = options->model_path;
            o.min_detection_confidence = options->min_detection_confidence;
            o.min_suppression_threshold = options->min_suppression_threshold;
            o.num_faces = options->max_faces;
            return new mlxpipe_face_detector{mlxpipe::FaceDetector::Create(o)};
        },
        static_cast<mlxpipe_face_detector*>(nullptr));
}

void mlxpipe_face_detector_destroy(mlxpipe_face_detector* detector) { delete detector; }

int mlxpipe_face_detector_detect(mlxpipe_face_detector* detector, const mlxpipe_image* image,
                                 mlxpipe_face_detection* out, int capacity) {
    return Guard(
        [&] {
            Require(detector, "detector");
            mlxpipe::FaceDetectorResult result = detector->impl->Detect(ToImage(image));
            const int n = result.detections.size();
            if (n > 0 && capacity > 0) Require(out, "out");
            for (int i = 0; i < std::min(n, capacity); ++i) {
                const mlxpipe::Detection& d = result.detections[i];
                out[i] = {d.score, d.box.xmin, d.box.ymin, d.box.width, d.box.height, {}};
                for (int k = 0; k < MLXPIPE_FACE_NUM_KEYPOINTS && k < int(d.keypoints.size()); ++k) {
                    out[i].keypoints[k] = {d.keypoints[k].x, d.keypoints[k].y};
                }
            }
            return n;
        },
        -1);
}

void mlxpipe_face_landmarker_default_options(mlxpipe_face_landmarker_options* options) {
    if (!options) return;
    mlxpipe::FaceLandmarkerOptions d;
    *options = {nullptr,
                d.num_faces,
                d.min_face_detection_confidence,
                d.min_face_presence_confidence,
                d.min_tracking_confidence,
                d.output_face_blendshapes,
                d.smooth_landmarks};
}

mlxpipe_face_landmarker* mlxpipe_face_landmarker_create(const mlxpipe_face_landmarker_options* options) {
    return Guard(
        [&]() -> mlxpipe_face_landmarker* {
            Require(options, "options");
            Require(options->model_path, "options->model_path");
            mlxpipe::FaceLandmarkerOptions o;
            o.model_asset_path = options->model_path;
            o.num_faces = options->max_faces;
            o.min_face_detection_confidence = options->min_face_detection_confidence;
            o.min_face_presence_confidence = options->min_face_presence_confidence;
            o.min_tracking_confidence = options->min_tracking_confidence;
            o.output_face_blendshapes = options->output_blendshapes != 0;
            o.smooth_landmarks = options->smooth_landmarks != 0;
            return new mlxpipe_face_landmarker{mlxpipe::FaceLandmarker::Create(o)};
        },
        static_cast<mlxpipe_face_landmarker*>(nullptr));
}

void mlxpipe_face_landmarker_destroy(mlxpipe_face_landmarker* landmarker) { delete landmarker; }

int mlxpipe_face_landmarker_detect(mlxpipe_face_landmarker* landmarker, const mlxpipe_image* image,
                                   mlxpipe_face_landmarks* out, int capacity) {
    return Guard(
        [&] {
            Require(landmarker, "landmarker");
            return CopyLandmarks(landmarker->impl->Detect(ToImage(image)), out, capacity);
        },
        -1);
}

int mlxpipe_face_landmarker_detect_for_video(mlxpipe_face_landmarker* landmarker, const mlxpipe_image* image,
                                             int64_t timestamp_ms, mlxpipe_face_landmarks* out, int capacity) {
    return Guard(
        [&] {
            Require(landmarker, "landmarker");
            return CopyLandmarks(landmarker->impl->DetectForVideo(ToImage(image), timestamp_ms), out, capacity);
        },
        -1);
}

const char* mlxpipe_face_blendshape_name(int index) {
    if (index < 0 || index >= int(mlxpipe::kFaceBlendshapeNames.size())) return nullptr;
    return mlxpipe::kFaceBlendshapeNames[index].c_str();
}

int mlxpipe_face_region_indices(mlxpipe_face_region region, const int** indices) {
    using namespace mlxpipe;
    static const std::vector<int> regions[MLXPIPE_FACE_NUM_REGIONS] = {
        RegionIndices(kFaceLandmarksLeftEye),     RegionIndices(kFaceLandmarksRightEye),
        RegionIndices(kFaceLandmarksLeftEyeBrow), RegionIndices(kFaceLandmarksRightEyeBrow),
        RegionIndices(kFaceLandmarksLeftIris),    RegionIndices(kFaceLandmarksRightIris),
        RegionIndices(kFaceLandmarksNose),        RegionIndices(kFaceLandmarksLips),
        RegionIndices(kFaceLandmarksFaceOval)};
    if (region < 0 || region >= MLXPIPE_FACE_NUM_REGIONS) {
        if (indices) *indices = nullptr;
        return 0;
    }
    if (indices) *indices = regions[region].data();
    return regions[region].size();
}

}  // extern "C"
