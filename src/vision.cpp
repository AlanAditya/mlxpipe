#include "src/vision.h"
#include "src/tensor_utils.h"
#include "src/ssd_anchors.h"
#include "src/tensors_to_detections.h"
#include "src/non_max_suppression.h"
#include "src/blaze_palm.h"
#include "src/blaze_hand.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <mlx/mlx.h>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cmath>
#include "src/one_euro_filter.h"

namespace mlx_vision {

static std::unique_ptr<BlazePalm> g_blaze_palm = nullptr;
static std::unique_ptr<BlazeHand> g_blaze_hand = nullptr;

struct HandTracker {
    bool active = false;
    std::vector<Landmark> prev_landmarks;
    std::unique_ptr<LandmarkOneEuroFilter> filter;
    
    HandTracker() {
        filter = std::make_unique<LandmarkOneEuroFilter>();
    }
    
    void reset() {
        active = false;
        filter->reset();
        prev_landmarks.clear();
    }
};

static const int MAX_HANDS = 2;
static std::vector<std::unique_ptr<HandTracker>> g_trackers;
static double g_timestamp = 0.0;

bool initialize_mlx(const std::string& palm_model_path, const std::string& hand_model_path) {
    try {
        auto palm_safetensors = mlx::core::load_safetensors(palm_model_path);
        g_blaze_palm = std::make_unique<BlazePalm>(palm_safetensors.first);
        
        auto hand_safetensors = mlx::core::load_safetensors(hand_model_path);
        g_blaze_hand = std::make_unique<BlazeHand>(hand_safetensors.first);
        
        for (int i = 0; i < MAX_HANDS; ++i) {
            g_trackers.push_back(std::make_unique<HandTracker>());
        }
        
        std::cout << "[MLX_VISION] Initialized MLX and loaded BlazePalm and BlazeHand" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[MLX_VISION] Error loading model: " << e.what() << std::endl;
        return false;
    }
}

static float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    cv::Rect2f inter = a & b;
    float inter_area = inter.area();
    float union_area = a.area() + b.area() - inter_area;
    return inter_area / (union_area + 1e-5f);
}

std::vector<HandResult> detect_hands(const cv::Mat& image) {
    auto t_start = std::chrono::high_resolution_clock::now();
    std::vector<HandResult> final_results;
    std::vector<cv::Rect2f> tracked_rects; // To compute IoU with Palm Detections
    
    g_timestamp += 1.0 / 30.0; // Simulate 30fps
    
    // 1. Preprocessing: Pad to square to preserve aspect ratio, then resize to 192x192 (for palm)
    int max_dim = std::max(image.cols, image.rows);
    cv::Mat padded_image = cv::Mat::zeros(max_dim, max_dim, image.type());
    int pad_offset_x = (max_dim - image.cols) / 2;
    int pad_offset_y = (max_dim - image.rows) / 2;
    image.copyTo(padded_image(cv::Rect(pad_offset_x, pad_offset_y, image.cols, image.rows)));
    
    cv::Mat resized_image;
    cv::resize(padded_image, resized_image, cv::Size(192, 192));
    
    auto t_preprocess = std::chrono::high_resolution_clock::now();
    double preprocess_ms = std::chrono::duration<double, std::milli>(t_preprocess - t_start).count();
    
    std::vector<HandResult> temp_results;
    std::vector<cv::Rect2f> temp_rects;
    std::vector<int> tracker_indices;
    
    double tracking_ms = 0.0;
    
    // --- TRACKING LOOP OVER ACTIVE TRACKERS ---
    for (int t_idx = 0; t_idx < g_trackers.size(); ++t_idx) {
        auto& tracker = g_trackers[t_idx];
        if (!tracker->active || tracker->prev_landmarks.size() != 21) {
            tracker->reset();
            continue;
        }
        
        // MediaPipe uses a weighted average of MCP joints for the rotation target.
        float x0 = tracker->prev_landmarks[0].x * image.cols;
        float y0 = tracker->prev_landmarks[0].y * image.rows;
        
        float x1 = (tracker->prev_landmarks[5].x + tracker->prev_landmarks[13].x) / 2.0f;
        float y1 = (tracker->prev_landmarks[5].y + tracker->prev_landmarks[13].y) / 2.0f;
        x1 = (x1 + tracker->prev_landmarks[9].x) / 2.0f * image.cols;
        y1 = (y1 + tracker->prev_landmarks[9].y) / 2.0f * image.rows;
        
        // MediaPipe formula: NormalizeRadians(kTargetAngle - std::atan2(-(y1 - y0), x1 - x0));
        // kTargetAngle = M_PI / 2 (90 degrees).
        float angle_rad = M_PI * 0.5f - std::atan2(-(y1 - y0), x1 - x0);
        // Normalize radians to [-pi, pi]
        angle_rad = angle_rad - 2 * M_PI * std::floor((angle_rad - (-M_PI)) / (2 * M_PI));
        float rotation = angle_rad * 180.0f / M_PI;
        
        float reverse_angle = -rotation * M_PI / 180.0f;
        
        // MediaPipe computes bounding box on a PARTIAL set of 12 landmarks (excluding fingertips and DIP joints).
        static constexpr int partial_indices[] = {0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18};
        
        // Find axis-aligned boundaries of partial landmarks
        float aa_min_x = 1e9, aa_max_x = -1e9, aa_min_y = 1e9, aa_max_y = -1e9;
        for (int idx : partial_indices) {
            float x = tracker->prev_landmarks[idx].x * image.cols;
            float y = tracker->prev_landmarks[idx].y * image.rows;
            aa_min_x = std::min(aa_min_x, x);
            aa_max_x = std::max(aa_max_x, x);
            aa_min_y = std::min(aa_min_y, y);
            aa_max_y = std::max(aa_max_y, y);
        }
        float axis_aligned_center_x = (aa_max_x + aa_min_x) / 2.0f;
        float axis_aligned_center_y = (aa_max_y + aa_min_y) / 2.0f;
        
        float min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        
        for (int idx : partial_indices) {
            float x = tracker->prev_landmarks[idx].x * image.cols - axis_aligned_center_x;
            float y = tracker->prev_landmarks[idx].y * image.rows - axis_aligned_center_y;
            float proj_x = x * std::cos(reverse_angle) - y * std::sin(reverse_angle);
            float proj_y = x * std::sin(reverse_angle) + y * std::cos(reverse_angle);
            min_x = std::min(min_x, proj_x);
            max_x = std::max(max_x, proj_x);
            min_y = std::min(min_y, proj_y);
            max_y = std::max(max_y, proj_y);
        }
        
        float proj_cx = (max_x + min_x) / 2.0f;
        float proj_cy = (max_y + min_y) / 2.0f;
        
        float cx = proj_cx * std::cos(rotation * M_PI / 180.0f) - proj_cy * std::sin(rotation * M_PI / 180.0f) + axis_aligned_center_x;
        float cy = proj_cx * std::sin(rotation * M_PI / 180.0f) + proj_cy * std::cos(rotation * M_PI / 180.0f) + axis_aligned_center_y;
        
        float width = max_x - min_x;
        float height = max_y - min_y;
        float crop_size = std::max(width, height) * 2.0f;
        
        float shift_y = -0.1f * height;
        cx += shift_y * std::sin(-rotation * M_PI / 180.0f);
        cy += shift_y * std::cos(-rotation * M_PI / 180.0f);
        
        if (crop_size <= 0.0f || std::isnan(crop_size)) {
            tracker->reset();
            continue;
        }

        // Save the expanded ROI bounding box to tracked_rects for IoU calculation with palm detections.
        // Both cx/cy and crop_size are in original image pixels.
        cv::Rect2f current_tracker_roi(cx - crop_size / 2.0f, cy - crop_size / 2.0f, crop_size, crop_size);

        cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), rotation, 1.0);
        float scale = 224.0f / crop_size;
        M.at<double>(0, 0) *= scale;
        M.at<double>(0, 1) *= scale;
        M.at<double>(1, 0) *= scale;
        M.at<double>(1, 1) *= scale;
        M.at<double>(0, 2) = M.at<double>(0, 2) * scale + 224.0 / 2.0 - cx * scale;
        M.at<double>(1, 2) = M.at<double>(1, 2) * scale + 224.0 / 2.0 - cy * scale;
        
        auto t_track_start = std::chrono::high_resolution_clock::now();
        
        cv::Mat hand_resized;
        cv::warpAffine(image, hand_resized, M, cv::Size(224, 224), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);
        if (M_inv.empty()) {
            tracker->reset();
            continue;
        }
        
        mlx::core::array hand_tensor = cvMat_to_mlxArray(hand_resized);
        hand_tensor = mlx::core::expand_dims(hand_tensor, 0);
        
        auto hand_out = g_blaze_hand->forward(hand_tensor);
        for (auto& arr : hand_out) mlx::core::eval(arr);
        
        auto t_track_end = std::chrono::high_resolution_clock::now();
        tracking_ms += std::chrono::duration<double, std::milli>(t_track_end - t_track_start).count();
        
        mlx::core::array landmarks_array = mlx::core::squeeze(hand_out[0]);
        float hand_presence_score = mlx::core::squeeze(hand_out[1]).item<float>();
        
        if (hand_presence_score < 0.5f) {
            tracker->reset();
            continue;
        }
        
        mlx::core::eval(landmarks_array);
        const float* lm_ptr = landmarks_array.data<float>();
        
        HandResult result;
        result.score = hand_presence_score;
        std::vector<float> xs(21), ys(21);
        
        float min_orig_x = 1e9, max_orig_x = -1e9, min_orig_y = 1e9, max_orig_y = -1e9;
        for (int i = 0; i < 21; ++i) {
            float lx_crop = lm_ptr[i * 3 + 0];
            float ly_crop = lm_ptr[i * 3 + 1];
            float lz_crop = lm_ptr[i * 3 + 2];
            
            float lx_orig = M_inv.at<double>(0, 0) * lx_crop + M_inv.at<double>(0, 1) * ly_crop + M_inv.at<double>(0, 2);
            float ly_orig = M_inv.at<double>(1, 0) * lx_crop + M_inv.at<double>(1, 1) * ly_crop + M_inv.at<double>(1, 2);
            
            xs[i] = lx_orig / (float)image.cols;
            ys[i] = ly_orig / (float)image.rows;
            
            Landmark lm;
            lm.x = xs[i];
            lm.y = ys[i];
            lm.z = lz_crop / 224.0f;
            result.landmarks.push_back(lm);
        }
        
        tracker->filter->filter(xs, ys, g_timestamp);
        
        for (int i = 0; i < 21; ++i) {
            result.landmarks[i].x = xs[i];
            result.landmarks[i].y = ys[i];
        }
        
        tracker->prev_landmarks = result.landmarks;
        
        temp_rects.push_back(current_tracker_roi);
        temp_results.push_back(result);
        tracker_indices.push_back(t_idx);
    }
    
    // Tracker NMS: If two trackers collapsed onto the same hand, keep the one with higher score
    std::vector<bool> keep(temp_results.size(), true);
    for (size_t i = 0; i < temp_results.size(); ++i) {
        if (!keep[i]) continue;
        cv::Point2f center_i(temp_rects[i].x + temp_rects[i].width / 2.0f, temp_rects[i].y + temp_rects[i].height / 2.0f);
        for (size_t j = i + 1; j < temp_results.size(); ++j) {
            if (!keep[j]) continue;
            cv::Point2f center_j(temp_rects[j].x + temp_rects[j].width / 2.0f, temp_rects[j].y + temp_rects[j].height / 2.0f);
            
            // If centers are inside each other's rects, they are tracking the same hand
            if (temp_rects[i].contains(center_j) || temp_rects[j].contains(center_i) || computeIoU(temp_rects[i], temp_rects[j]) > 0.1f) {
                if (temp_results[i].score >= temp_results[j].score) {
                    keep[j] = false;
                } else {
                    keep[i] = false;
                    break;
                }
            }
        }
    }
    
    for (size_t i = 0; i < temp_results.size(); ++i) {
        if (keep[i]) {
            tracked_rects.push_back(temp_rects[i]);
            final_results.push_back(temp_results[i]);
        } else {
            g_trackers[tracker_indices[i]]->reset();
        }
    }
    
    auto t_nms = std::chrono::high_resolution_clock::now();
    double nms_ms = std::chrono::duration<double, std::milli>(t_nms - t_preprocess).count() - tracking_ms;
    
    double palm_ms = 0.0;
    
    // 2. If we need more hands, run BlazePalm
    if (final_results.size() < MAX_HANDS && g_blaze_palm) {
        auto t_palm_start = std::chrono::high_resolution_clock::now();
        SsdAnchorsOptions anchor_options;
        anchor_options.input_size_width = 192;
        anchor_options.input_size_height = 192;
        anchor_options.min_scale = 0.1484375f;
        anchor_options.max_scale = 0.75f;
        anchor_options.num_layers = 4;
        anchor_options.strides = {8, 16, 16, 16};
        anchor_options.aspect_ratios = {1.0f};
        anchor_options.reduce_boxes_in_lowest_layer = false;
        anchor_options.interpolated_scale_aspect_ratio = 1.0f;
        anchor_options.fixed_anchor_size = true;
        std::vector<Anchor> anchors = GenerateSsdAnchors(anchor_options);
        
        mlx::core::array input_tensor = cvMat_to_mlxArray(resized_image);
        input_tensor = mlx::core::expand_dims(input_tensor, 0);
        
        auto net_out = g_blaze_palm->forward(input_tensor);
        for (auto& arr : net_out) mlx::core::eval(arr);
        
        auto t_palm_end = std::chrono::high_resolution_clock::now();
        palm_ms += std::chrono::duration<double, std::milli>(t_palm_end - t_palm_start).count();
        
        mlx::core::array box_tensor = mlx::core::squeeze(net_out[0], 0);
        mlx::core::array score_tensor = mlx::core::squeeze(net_out[1], 0);
        if (box_tensor.shape(1) == 1) {
            auto temp = box_tensor; box_tensor = score_tensor; score_tensor = temp;
        }
        
        TensorsToDetectionsOptions t2d_options;
        t2d_options.num_classes = 1;
        t2d_options.num_boxes = anchors.size();
        t2d_options.num_coords = 18;
        t2d_options.num_keypoints = 7;
        t2d_options.x_scale = 192.0f;
        t2d_options.y_scale = 192.0f;
        t2d_options.w_scale = 192.0f;
        t2d_options.h_scale = 192.0f;
        t2d_options.min_score_thresh = 0.5f;
        t2d_options.reverse_output_order = true;
        
        std::vector<Detection> palms = DecodeTensorsToDetections(box_tensor, score_tensor, anchors, t2d_options);
        NmsOptions nms_options;
        nms_options.min_suppression_threshold = 0.3f;
        nms_options.min_score_threshold = 0.5f;
        std::vector<Detection> nms_palms = ApplyNMS(palms, nms_options);
        
        for (const auto& det : nms_palms) {
            if (final_results.size() >= MAX_HANDS) break;
            
            float cx_palm = (det.x + det.width / 2.0f) * max_dim - pad_offset_x;
            float cy_palm = (det.y + det.height / 2.0f) * max_dim - pad_offset_y;
            float w_palm = det.width * max_dim;
            float h_palm = det.height * max_dim;
            float kp0_x = det.keypoints[0].first * max_dim - pad_offset_x;
            float kp0_y = det.keypoints[0].second * max_dim - pad_offset_y;
            float kp2_x = det.keypoints[2].first * max_dim - pad_offset_x;
            float kp2_y = det.keypoints[2].second * max_dim - pad_offset_y;
            
            float angle = std::atan2(kp2_y - kp0_y, kp2_x - kp0_x);
            float rotation = angle * 180.0f / M_PI + 90.0f;
            
            float dx = det.width * max_dim * 0.5f * std::sin(rotation * M_PI / 180.0f);
            float dy = -det.height * max_dim * 0.5f * std::cos(rotation * M_PI / 180.0f);
            float cx = cx_palm + dx;
            float cy = cy_palm + dy;
            
            float crop_size = std::max(det.width, det.height) * max_dim * 2.6f;
            if (crop_size <= 0.0f || std::isnan(crop_size)) continue;
            
            // Now compute IoU between the expanded palm ROI and the tracked ROIs
            cv::Rect2f new_rect(cx - crop_size / 2.0f, cy - crop_size / 2.0f, crop_size, crop_size);
            cv::Point2f palm_center(cx, cy);
            
            bool overlap = false;
            for (const auto& tr : tracked_rects) {
                // If the palm center is inside the existing tracked hand ROI, it's the same hand
                if (tr.contains(palm_center) || computeIoU(new_rect, tr) > 0.1f) {
                    overlap = true;
                    break;
                }
            }
            if (overlap) continue;
            
            cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(cx, cy), rotation, 1.0);
            float scale = 224.0f / crop_size;
            M.at<double>(0, 0) *= scale;
            M.at<double>(0, 1) *= scale;
            M.at<double>(1, 0) *= scale;
            M.at<double>(1, 1) *= scale;
            M.at<double>(0, 2) = M.at<double>(0, 2) * scale + 224.0 / 2.0 - cx * scale;
            M.at<double>(1, 2) = M.at<double>(1, 2) * scale + 224.0 / 2.0 - cy * scale;
            
            cv::Mat hand_resized;
            cv::warpAffine(image, hand_resized, M, cv::Size(224, 224), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            cv::Mat M_inv;
            cv::invertAffineTransform(M, M_inv);
            if (M_inv.empty()) continue;
            
            mlx::core::array hand_tensor = cvMat_to_mlxArray(hand_resized);
            hand_tensor = mlx::core::expand_dims(hand_tensor, 0);
            
            auto hand_out = g_blaze_hand->forward(hand_tensor);
            for (auto& arr : hand_out) mlx::core::eval(arr);
            
            mlx::core::array landmarks_array = mlx::core::squeeze(hand_out[0]);
            float hand_presence_score = mlx::core::squeeze(hand_out[1]).item<float>();
            
            if (hand_presence_score < 0.5f) continue;
            
            mlx::core::eval(landmarks_array);
            const float* lm_ptr = landmarks_array.data<float>();
            
            HandResult result;
            result.score = hand_presence_score;
            std::vector<float> xs(21), ys(21);
            
            for (int i = 0; i < 21; ++i) {
                float lx_crop = lm_ptr[i * 3 + 0];
                float ly_crop = lm_ptr[i * 3 + 1];
                float lz_crop = lm_ptr[i * 3 + 2];
                
                float lx_orig = M_inv.at<double>(0, 0) * lx_crop + M_inv.at<double>(0, 1) * ly_crop + M_inv.at<double>(0, 2);
                float ly_orig = M_inv.at<double>(1, 0) * lx_crop + M_inv.at<double>(1, 1) * ly_crop + M_inv.at<double>(1, 2);
                
                xs[i] = lx_orig / (float)image.cols;
                ys[i] = ly_orig / (float)image.rows;
                
                Landmark lm;
                lm.x = xs[i];
                lm.y = ys[i];
                lm.z = lz_crop / 224.0f;
                result.landmarks.push_back(lm);
            }
            
            // Find an inactive tracker and assign
            for (auto& tracker : g_trackers) {
                if (!tracker->active) {
                    tracker->active = true;
                    tracker->filter->reset();
                    tracker->filter->filter(xs, ys, g_timestamp);
                    for (int i = 0; i < 21; ++i) {
                        result.landmarks[i].x = xs[i];
                        result.landmarks[i].y = ys[i];
                    }
                    tracker->prev_landmarks = result.landmarks;
                    break;
                }
            }
            
            final_results.push_back(result);
        }
    }
    
    auto t_final = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_final - t_start).count();
    
    // Only print profiling if we actually did tracking or palm detection
    if (tracking_ms > 0 || palm_ms > 0) {
        std::cout << "[PROFILE] Preprocess: " << preprocess_ms << "ms | Tracking (" << g_trackers.size() << "): " << tracking_ms << "ms | Palm: " << palm_ms << "ms | Total: " << total_ms << "ms\n";
    }
    
    return final_results;
}

} // namespace mlx_vision
