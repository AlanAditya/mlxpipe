#include "src/vision.h"
#include "src/ssd_anchors.h"
#include "src/tensors_to_detections.h"
#include "src/non_max_suppression.h"
#include "src/blaze_palm.h"
#include "src/blaze_hand.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/geometry/2d.hpp>
#include <mlx/mlx.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include "src/one_euro_filter.h"

// Per-frame flow mirrors MediaPipe's HandLandmarkerGraph in stream mode
// (mediapipe/tasks/cc/vision/hand_landmarker/hand_landmarker_graph.cc):
//   1. Hand rects predicted from the previous frame's landmarks are reused.
//   2. Palm detection runs only while fewer than MAX_HANDS hands are tracked.
//   3. Palm-derived hand rects are kept only if their IoU with every tracked
//      rect is <= kMinTrackingConfidence (HandAssociationCalculator).
//   4. Landmarks run on every rect; hands below kMinHandPresence are dropped.
//   5. Duplicate hands are removed by comparing landmarks
//      (HandLandmarksDeduplicationCalculator).
//   6. Next-frame rects come from the raw (unsmoothed) landmarks.
// All GPU work for the tracked hands and palm detection is submitted together
// so each frame pays for a single GPU round trip.

namespace mlx_vision {

namespace mx = mlx::core;

static constexpr int MAX_HANDS = 2;
static constexpr int kPalmInputSize = 192;
static constexpr int kHandInputSize = 224;
static constexpr int kNumLandmarks = 21;
// MediaPipe HandLandmarker defaults.
static constexpr float kMinDetectionConfidence = 0.5f;
static constexpr float kMinHandPresence = 0.5f;
static constexpr float kMinTrackingConfidence = 0.5f;

// A square region of the image in pixels, rotated clockwise by `rotation`
// radians around its centre: MediaPipe's NormalizedRect after square_long.
struct HandRect {
    float cx = 0.0f;
    float cy = 0.0f;
    float size = 0.0f;
    float rotation = 0.0f;
    int id = 0;
};

struct HandCandidate {
    HandResult result;  // raw landmarks, image-normalized
    HandRect rect;      // the rect the landmarks were computed from
};

static std::unique_ptr<BlazePalm> g_blaze_palm = nullptr;
static std::unique_ptr<BlazeHand> g_blaze_hand = nullptr;
static std::function<std::vector<mx::array>(const std::vector<mx::array>&)> g_run_models;
static std::vector<Anchor> g_anchors;

static std::vector<HandRect> g_prev_rects;  // HAND_RECT_NEXT_FRAME of the previous frame
static std::map<int, std::unique_ptr<LandmarkOneEuroFilter>> g_filters;  // keyed by rect id
static int g_next_rect_id = 1;
static std::chrono::steady_clock::time_point g_start_time;

// Runs each input through the model matching its size. Inputs are uint8 RGB
// NHWC batches; scaling to [0, 1] happens on the GPU.
static std::vector<mx::array> run_models(const std::vector<mx::array>& inputs) {
    std::vector<mx::array> outputs;
    for (const auto& in : inputs) {
        auto x = mx::divide(mx::astype(in, mx::float32), mx::array(255.0f));
        auto out = in.shape(1) == kHandInputSize ? g_blaze_hand->forward(x) : g_blaze_palm->forward(x);
        outputs.insert(outputs.end(), out.begin(), out.end());
    }
    return outputs;
}

bool initialize_mlx(const std::string& palm_model_path, const std::string& hand_model_path) {
    try {
        auto palm_safetensors = mx::load_safetensors(palm_model_path);
        g_blaze_palm = std::make_unique<BlazePalm>(palm_safetensors.first);

        auto hand_safetensors = mx::load_safetensors(hand_model_path);
        g_blaze_hand = std::make_unique<BlazeHand>(hand_safetensors.first);

        // Fuses the elementwise ops (pad, bias, ReLU6, PReLU) between convolutions.
        g_run_models = mx::compile(run_models);
        // Trace every input combination detect_hands submits so the first
        // frames don't stall on compilation.
        auto palm_in = mx::zeros({1, kPalmInputSize, kPalmInputSize, 3}, mx::uint8);
        for (int n = 1; n <= MAX_HANDS; ++n) {
            auto hands_in = mx::zeros({n, kHandInputSize, kHandInputSize, 3}, mx::uint8);
            mx::eval(g_run_models({hands_in}));
            mx::eval(g_run_models({hands_in, palm_in}));
        }
        mx::eval(g_run_models({palm_in}));

        // MediaPipe hand_detector_graph.cc ConfigureSsdAnchorsCalculator.
        SsdAnchorsOptions anchor_options;
        anchor_options.input_size_width = kPalmInputSize;
        anchor_options.input_size_height = kPalmInputSize;
        anchor_options.min_scale = 0.1484375f;
        anchor_options.max_scale = 0.75f;
        anchor_options.num_layers = 4;
        anchor_options.strides = {8, 16, 16, 16};
        anchor_options.aspect_ratios = {1.0f};
        anchor_options.reduce_boxes_in_lowest_layer = false;
        anchor_options.interpolated_scale_aspect_ratio = 1.0f;
        anchor_options.fixed_anchor_size = true;
        g_anchors = GenerateSsdAnchors(anchor_options);

        g_prev_rects.clear();
        g_filters.clear();
        g_start_time = std::chrono::steady_clock::now();

        std::cout << "[MLX_VISION] Initialized MLX and loaded BlazePalm and BlazeHand" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MLX_VISION] Error loading model: " << e.what() << std::endl;
        return false;
    }
}

static float normalize_radians(float angle) {
    return angle - 2 * M_PI * std::floor((angle - (-M_PI)) / (2 * M_PI));
}

// Axis-aligned IoU ignoring rotation, like MediaPipe's DoesRectOverlap.
static float rect_iou(const HandRect& a, const HandRect& b) {
    cv::Rect2f ra(a.cx - a.size / 2.0f, a.cy - a.size / 2.0f, a.size, a.size);
    cv::Rect2f rb(b.cx - b.size / 2.0f, b.cy - b.size / 2.0f, b.size, b.size);
    float inter = (ra & rb).area();
    float uni = ra.area() + rb.area() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// RectTransformationCalculator with shift_x = 0 and square_long, in pixels.
static HandRect transform_rect(float cx, float cy, float width, float height, float rotation,
                               float shift_y, float scale) {
    HandRect rect;
    rect.cx = cx - height * shift_y * std::sin(rotation);
    rect.cy = cy + height * shift_y * std::cos(rotation);
    rect.size = std::max(width, height) * scale;
    rect.rotation = rotation;
    return rect;
}

// DetectionsToRectsCalculator (rotation from wrist to middle-finger MCP,
// target 90 degrees) followed by RectTransformation(scale 2.6, shift_y -0.5).
// Detection coordinates are normalized to the letterboxed square.
static HandRect palm_to_hand_rect(const Detection& det, float max_dim, float pad_x, float pad_y) {
    float cx = (det.x + det.width / 2.0f) * max_dim - pad_x;
    float cy = (det.y + det.height / 2.0f) * max_dim - pad_y;
    float x0 = det.keypoints[0].first * max_dim - pad_x;
    float y0 = det.keypoints[0].second * max_dim - pad_y;
    float x1 = det.keypoints[2].first * max_dim - pad_x;
    float y1 = det.keypoints[2].second * max_dim - pad_y;
    float rotation = normalize_radians(M_PI * 0.5f - std::atan2(-(y1 - y0), x1 - x0));
    return transform_rect(cx, cy, det.width * max_dim, det.height * max_dim, rotation, -0.5f, 2.6f);
}

// HandLandmarksToRectCalculator on the partial landmark set used by
// hand_landmark_landmarks_to_roi.pbtxt, followed by
// RectTransformation(scale 2.0, shift_y -0.1).
static HandRect landmarks_to_hand_rect(const std::vector<Landmark>& lms, int image_w, int image_h) {
    static constexpr int partial_indices[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};

    // Rotation from the wrist to the average of the index, middle and ring MCPs.
    float x0 = lms[0].x * image_w;
    float y0 = lms[0].y * image_h;
    float x1 = ((lms[5].x + lms[13].x) / 2.0f + lms[9].x) / 2.0f * image_w;
    float y1 = ((lms[5].y + lms[13].y) / 2.0f + lms[9].y) / 2.0f * image_h;
    float rotation = normalize_radians(M_PI * 0.5f - std::atan2(-(y1 - y0), x1 - x0));
    float reverse_angle = -rotation;

    float aa_min_x = 1e9, aa_max_x = -1e9, aa_min_y = 1e9, aa_max_y = -1e9;
    for (int idx : partial_indices) {
        aa_min_x = std::min(aa_min_x, lms[idx].x * image_w);
        aa_max_x = std::max(aa_max_x, lms[idx].x * image_w);
        aa_min_y = std::min(aa_min_y, lms[idx].y * image_h);
        aa_max_y = std::max(aa_max_y, lms[idx].y * image_h);
    }
    float aa_cx = (aa_max_x + aa_min_x) / 2.0f;
    float aa_cy = (aa_max_y + aa_min_y) / 2.0f;

    // Bounds of the landmarks in the hand-aligned frame.
    float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (int idx : partial_indices) {
        float x = lms[idx].x * image_w - aa_cx;
        float y = lms[idx].y * image_h - aa_cy;
        float proj_x = x * std::cos(reverse_angle) - y * std::sin(reverse_angle);
        float proj_y = x * std::sin(reverse_angle) + y * std::cos(reverse_angle);
        min_x = std::min(min_x, proj_x);
        max_x = std::max(max_x, proj_x);
        min_y = std::min(min_y, proj_y);
        max_y = std::max(max_y, proj_y);
    }
    float proj_cx = (max_x + min_x) / 2.0f;
    float proj_cy = (max_y + min_y) / 2.0f;
    float cx = proj_cx * std::cos(rotation) - proj_cy * std::sin(rotation) + aa_cx;
    float cy = proj_cx * std::sin(rotation) + proj_cy * std::cos(rotation) + aa_cy;

    return transform_rect(cx, cy, max_x - min_x, max_y - min_y, rotation, -0.1f, 2.0f);
}

// Affine map from image pixels to the 224x224 crop of `rect`.
static cv::Mat crop_matrix(const HandRect& rect) {
    float scale = kHandInputSize / rect.size;
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(rect.cx, rect.cy), rect.rotation * 180.0f / M_PI, scale);
    M.at<double>(0, 2) += kHandInputSize / 2.0 - rect.cx;
    M.at<double>(1, 2) += kHandInputSize / 2.0 - rect.cy;
    return M;
}

// Crops every rect into one uint8 RGB batch [n, 224, 224, 3]. Fills the
// inverse (crop -> image) matrices used to project landmarks back.
static mx::array make_hand_batch(const cv::Mat& image, const std::vector<HandRect>& rects,
                                 std::vector<cv::Mat>& inverse_matrices) {
    const int n = rects.size();
    cv::Mat batch(n * kHandInputSize, kHandInputSize, CV_8UC3);
    cv::Mat crop;
    for (int i = 0; i < n; ++i) {
        cv::Mat M = crop_matrix(rects[i]);
        // ImageToTensorCalculator defaults to BORDER_REPLICATE.
        cv::warpAffine(image, crop, M, cv::Size(kHandInputSize, kHandInputSize), cv::INTER_LINEAR,
                       cv::BORDER_REPLICATE);
        cv::cvtColor(crop, batch.rowRange(i * kHandInputSize, (i + 1) * kHandInputSize), cv::COLOR_BGR2RGB);
        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);
        inverse_matrices.push_back(M_inv);
    }
    return mx::array(batch.data, {n, kHandInputSize, kHandInputSize, 3}, mx::uint8);
}

// Letterboxes the frame into a centred 192x192 square with zero borders, like
// ImageToTensorCalculator with keep_aspect_ratio and BORDER_ZERO.
static mx::array make_palm_input(const cv::Mat& image, float& max_dim, float& pad_x, float& pad_y) {
    max_dim = std::max(image.cols, image.rows);
    pad_x = (max_dim - image.cols) / 2.0f;
    pad_y = (max_dim - image.rows) / 2.0f;
    float scale = kPalmInputSize / max_dim;
    cv::Mat M = (cv::Mat_<double>(2, 3) << scale, 0, pad_x * scale, 0, scale, pad_y * scale);
    cv::Mat letterboxed, rgb;
    cv::warpAffine(image, letterboxed, M, cv::Size(kPalmInputSize, kPalmInputSize), cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    cv::cvtColor(letterboxed, rgb, cv::COLOR_BGR2RGB);
    return mx::array(rgb.data, {1, kPalmInputSize, kPalmInputSize, 3}, mx::uint8);
}

// Projects BlazeHand outputs for batch entry `i` back to the image.
// Outputs: [0] n x 63 landmarks in crop pixels, [1] hand presence,
// [2] handedness (MediaPipe labels > 0.5 as "Left"), [3] n x 63 world
// landmarks in metres, oriented like the crop.
static HandResult decode_hand_output(const std::vector<mx::array>& hand_out, int i, const HandRect& rect,
                                     const cv::Mat& M_inv, int image_w, int image_h) {
    const float* lm_ptr = hand_out[0].data<float>() + i * kNumLandmarks * 3;
    const float* world_ptr = hand_out[3].data<float>() + i * kNumLandmarks * 3;

    double a = M_inv.at<double>(0, 0), b = M_inv.at<double>(0, 1), tx = M_inv.at<double>(0, 2);
    double c = M_inv.at<double>(1, 0), d = M_inv.at<double>(1, 1), ty = M_inv.at<double>(1, 2);
    // M_inv's linear part is a rotation scaled by crop_size / 224; normalize it
    // to rotate world landmarks back into image orientation.
    double inv_scale = std::sqrt(a * a + c * c);

    HandResult result;
    result.score = hand_out[1].data<float>()[i];
    result.is_left_hand = hand_out[2].data<float>()[i] > 0.5f;

    for (int k = 0; k < kNumLandmarks; ++k) {
        float lx_crop = lm_ptr[k * 3 + 0];
        float ly_crop = lm_ptr[k * 3 + 1];
        float lz_crop = lm_ptr[k * 3 + 2];

        Landmark lm;
        lm.x = (a * lx_crop + b * ly_crop + tx) / image_w;
        lm.y = (c * lx_crop + d * ly_crop + ty) / image_h;
        // TensorsToLandmarks (normalize_z 0.4) then LandmarkProjection (z * rect width).
        lm.z = lz_crop / kHandInputSize / 0.4f * (rect.size / image_w);
        result.landmarks.push_back(lm);

        float wx = world_ptr[k * 3 + 0];
        float wy = world_ptr[k * 3 + 1];
        Landmark wlm;
        wlm.x = (a * wx + b * wy) / inv_scale;
        wlm.y = (c * wx + d * wy) / inv_scale;
        wlm.z = world_ptr[k * 3 + 2];
        result.world_landmarks.push_back(wlm);
    }
    return result;
}

// Appends the hands from an evaluated BlazeHand batch whose presence passes.
static void collect_hands(const std::vector<mx::array>& hand_out, const std::vector<HandRect>& rects,
                          const std::vector<cv::Mat>& inverse_matrices, int image_w, int image_h,
                          std::vector<HandCandidate>& candidates) {
    for (size_t i = 0; i < rects.size(); ++i) {
        if (hand_out[1].data<float>()[i] < kMinHandPresence) continue;
        candidates.push_back({decode_hand_output(hand_out, i, rects[i], inverse_matrices[i], image_w, image_h),
                              rects[i]});
    }
}

static float landmark_distance(const Landmark& a, const Landmark& b, int image_w, int image_h) {
    return std::hypot((a.x - b.x) * image_w, (a.y - b.y) * image_h);
}

static cv::Rect2f landmark_bounds(const std::vector<Landmark>& lms) {
    float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
    for (const auto& lm : lms) {
        min_x = std::min(min_x, lm.x);
        max_x = std::max(max_x, lm.x);
        min_y = std::min(min_y, lm.y);
        max_y = std::max(max_y, lm.y);
    }
    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

// Port of HandLandmarksDeduplicationCalculator: a hand is a duplicate of an
// earlier kept hand when at least 10 of its 21 landmarks lie within 20% of the
// larger palm size and their landmark bounding boxes have IoU > 0.2.
static std::vector<HandCandidate> deduplicate_hands(const std::vector<HandCandidate>& hands, int image_w,
                                                    int image_h) {
    std::vector<float> baselines;
    std::vector<cv::Rect2f> bounds;
    for (const auto& hand : hands) {
        const auto& lms = hand.result.landmarks;
        baselines.push_back(std::max({landmark_distance(lms[0], lms[5], image_w, image_h),
                                      landmark_distance(lms[5], lms[17], image_w, image_h),
                                      landmark_distance(lms[17], lms[0], image_w, image_h)}));
        bounds.push_back(landmark_bounds(lms));
    }

    std::vector<int> retained;
    for (int i = 0; i < (int)hands.size(); ++i) {
        bool suppressed = false;
        for (int j : retained) {
            float threshold = std::max(baselines[i], baselines[j]) * 0.2f;
            int matched = 0;
            for (int k = 0; k < kNumLandmarks; ++k) {
                if (landmark_distance(hands[i].result.landmarks[k], hands[j].result.landmarks[k], image_w,
                                      image_h) < threshold) {
                    ++matched;
                }
            }
            float inter = (bounds[i] & bounds[j]).area();
            float uni = bounds[i].area() + bounds[j].area() - inter;
            float iou = uni > 0.0f ? inter / uni : 0.0f;
            if (matched >= 10 && iou > 0.2f) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) retained.push_back(i);
    }

    std::vector<HandCandidate> kept;
    for (int i : retained) kept.push_back(hands[i]);
    return kept;
}

// Sets result.box to the normalized axis-aligned bounds of the landmarks.
static void set_landmark_box(HandResult& result) {
    cv::Rect2f bounds = landmark_bounds(result.landmarks);
    result.box.x = bounds.x;
    result.box.y = bounds.y;
    result.box.width = bounds.width;
    result.box.height = bounds.height;
    result.box.confidence = result.score;
}

// Runs the One Euro filter over the landmark x/y coordinates in place.
static void smooth_landmarks(LandmarkOneEuroFilter& filter, HandResult& result, double timestamp) {
    std::vector<float> xs(kNumLandmarks), ys(kNumLandmarks);
    for (int i = 0; i < kNumLandmarks; ++i) {
        xs[i] = result.landmarks[i].x;
        ys[i] = result.landmarks[i].y;
    }
    filter.filter(xs, ys, timestamp);
    for (int i = 0; i < kNumLandmarks; ++i) {
        result.landmarks[i].x = xs[i];
        result.landmarks[i].y = ys[i];
    }
}

std::vector<HandResult> detect_hands(const cv::Mat& image) {
    using clock = std::chrono::steady_clock;
    auto t_start = clock::now();
    const int image_w = image.cols;
    const int image_h = image.rows;
    double timestamp = std::chrono::duration<double>(t_start - g_start_time).count();

    // 1. Tracked hands from the previous frame; palm detection only when short of MAX_HANDS.
    std::vector<HandRect> tracked_rects = g_prev_rects;
    bool run_palm = tracked_rects.size() < MAX_HANDS && g_blaze_palm;

    std::vector<mx::array> inputs;
    std::vector<cv::Mat> tracked_inverse;
    if (!tracked_rects.empty()) inputs.push_back(make_hand_batch(image, tracked_rects, tracked_inverse));
    float max_dim = 0, pad_x = 0, pad_y = 0;
    if (run_palm) inputs.push_back(make_palm_input(image, max_dim, pad_x, pad_y));
    auto t_preprocess = clock::now();

    // 2. One GPU submission for the tracked hands and palm detection.
    std::vector<mx::array> outputs;
    if (!inputs.empty()) {
        outputs = g_run_models(inputs);
        mx::eval(outputs);
    }
    auto t_inference = clock::now();

    std::vector<HandCandidate> candidates;
    if (!tracked_rects.empty()) {
        std::vector<mx::array> hand_out(outputs.begin(), outputs.begin() + 4);
        collect_hands(hand_out, tracked_rects, tracked_inverse, image_w, image_h, candidates);
    }

    // 3. Palm detections -> hand rects, associated against the tracked rects.
    if (run_palm) {
        size_t palm_offset = tracked_rects.empty() ? 0 : 4;
        mx::array box_tensor = mx::squeeze(outputs[palm_offset], 0);
        mx::array score_tensor = mx::squeeze(outputs[palm_offset + 1], 0);
        if (box_tensor.shape(1) == 1) std::swap(box_tensor, score_tensor);

        // MediaPipe hand_detector_graph.cc ConfigureTensorsToDetectionsCalculator.
        TensorsToDetectionsOptions t2d_options;
        t2d_options.num_classes = 1;
        t2d_options.num_boxes = g_anchors.size();
        t2d_options.num_coords = 18;
        t2d_options.num_keypoints = 7;
        t2d_options.x_scale = kPalmInputSize;
        t2d_options.y_scale = kPalmInputSize;
        t2d_options.w_scale = kPalmInputSize;
        t2d_options.h_scale = kPalmInputSize;
        t2d_options.min_score_thresh = kMinDetectionConfidence;
        t2d_options.reverse_output_order = true;
        std::vector<Detection> palms = DecodeTensorsToDetections(box_tensor, score_tensor, g_anchors, t2d_options);

        NmsOptions nms_options;
        nms_options.min_suppression_threshold = 0.3f;
        nms_options.weighted = true;
        nms_options.max_num_detections = MAX_HANDS;
        std::vector<Detection> nms_palms = ApplyNMS(palms, nms_options);

        std::vector<HandRect> associated = tracked_rects;
        std::vector<HandRect> new_rects;
        for (const auto& det : nms_palms) {
            if (associated.size() >= MAX_HANDS) break;
            HandRect rect = palm_to_hand_rect(det, max_dim, pad_x, pad_y);
            if (rect.size <= 0.0f || std::isnan(rect.size)) continue;
            bool overlaps = false;
            for (const auto& existing : associated) {
                if (rect_iou(rect, existing) > kMinTrackingConfidence) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) continue;
            rect.id = g_next_rect_id++;
            associated.push_back(rect);
            new_rects.push_back(rect);
        }

        // Newly found hands need their own landmark pass this frame.
        if (!new_rects.empty()) {
            std::vector<cv::Mat> new_inverse;
            auto hand_out = g_run_models({make_hand_batch(image, new_rects, new_inverse)});
            mx::eval(hand_out);
            collect_hands(hand_out, new_rects, new_inverse, image_w, image_h, candidates);
        }
    }
    auto t_palm = clock::now();

    // 4. Drop duplicates, predict next-frame rects from raw landmarks, smooth the output.
    candidates = deduplicate_hands(candidates, image_w, image_h);

    std::vector<HandResult> final_results;
    std::vector<HandRect> next_rects;
    std::map<int, std::unique_ptr<LandmarkOneEuroFilter>> filters;
    for (auto& candidate : candidates) {
        HandRect next = landmarks_to_hand_rect(candidate.result.landmarks, image_w, image_h);
        if (next.size <= 0.0f || std::isnan(next.size)) continue;
        next.id = candidate.rect.id;
        next_rects.push_back(next);

        auto it = g_filters.find(next.id);
        auto filter = it != g_filters.end() ? std::move(it->second) : std::make_unique<LandmarkOneEuroFilter>();
        smooth_landmarks(*filter, candidate.result, timestamp);
        filters[next.id] = std::move(filter);

        set_landmark_box(candidate.result);
        final_results.push_back(candidate.result);
    }
    g_prev_rects = next_rects;
    g_filters = std::move(filters);

    auto t_final = clock::now();
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    if (!inputs.empty()) {
        std::cout << "[PROFILE] Preprocess: " << ms(t_start, t_preprocess) << "ms | Inference ("
                  << tracked_rects.size() << " tracked" << (run_palm ? " + palm" : "")
                  << "): " << ms(t_preprocess, t_inference) << "ms | Acquire: " << ms(t_inference, t_palm)
                  << "ms | Total: " << ms(t_start, t_final) << "ms | Hands: " << final_results.size() << "\n";
    }

    return final_results;
}

} // namespace mlx_vision
