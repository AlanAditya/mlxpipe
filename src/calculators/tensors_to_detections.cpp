#include "mlxpipe/calculators/tensors_to_detections.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mlxpipe {

namespace mx = mlx::core;

std::vector<Detection> DecodeDetections(const float* raw_boxes, const float* raw_scores,
                                        const std::vector<Anchor>& anchors,
                                        const TensorsToDetectionsOptions& options) {
    using BoxFormat = TensorsToDetectionsOptions::BoxFormat;
    BoxFormat format = options.box_format;
    if (format == BoxFormat::kUnspecified) format = options.reverse_output_order ? BoxFormat::kXYWH : BoxFormat::kYXHW;
    if (static_cast<int>(anchors.size()) != options.num_boxes) {
        throw std::invalid_argument("TensorsToDetections: anchor count does not match num_boxes");
    }

    std::vector<Detection> detections;
    for (int i = 0; i < options.num_boxes; ++i) {
        // Score: maximum over classes.
        float max_score = std::numeric_limits<float>::lowest();
        int class_id = -1;
        for (int c = 0; c < options.num_classes; ++c) {
            float score = raw_scores[i * options.num_classes + c];
            if (options.sigmoid_score) {
                if (options.score_clipping_thresh > 0) {
                    score = std::clamp(score, -options.score_clipping_thresh, options.score_clipping_thresh);
                }
                score = 1.0f / (1.0f + std::exp(-score));
            }
            if (max_score < score) {
                max_score = score;
                class_id = c;
            }
        }
        if (max_score < options.min_score_thresh) continue;

        const int box_offset = i * options.num_coords + options.box_coord_offset;
        float y_center = 0, x_center = 0, h = 0, w = 0;
        switch (format) {
            case BoxFormat::kUnspecified:
            case BoxFormat::kYXHW:
                y_center = raw_boxes[box_offset];
                x_center = raw_boxes[box_offset + 1];
                h = raw_boxes[box_offset + 2];
                w = raw_boxes[box_offset + 3];
                break;
            case BoxFormat::kXYWH:
                x_center = raw_boxes[box_offset];
                y_center = raw_boxes[box_offset + 1];
                w = raw_boxes[box_offset + 2];
                h = raw_boxes[box_offset + 3];
                break;
            case BoxFormat::kXYXY:
                x_center = (-raw_boxes[box_offset] + raw_boxes[box_offset + 2]) / 2;
                y_center = (-raw_boxes[box_offset + 1] + raw_boxes[box_offset + 3]) / 2;
                w = raw_boxes[box_offset + 2] + raw_boxes[box_offset];
                h = raw_boxes[box_offset + 3] + raw_boxes[box_offset + 1];
                break;
        }
        x_center = x_center / options.x_scale * anchors[i].w + anchors[i].x_center;
        y_center = y_center / options.y_scale * anchors[i].h + anchors[i].y_center;
        if (options.apply_exponential_on_box_size) {
            h = std::exp(h / options.h_scale) * anchors[i].h;
            w = std::exp(w / options.w_scale) * anchors[i].w;
        } else {
            h = h / options.h_scale * anchors[i].h;
            w = w / options.w_scale * anchors[i].w;
        }
        float ymin = y_center - h / 2.f;
        float xmin = x_center - w / 2.f;
        float ymax = y_center + h / 2.f;
        float xmax = x_center + w / 2.f;
        // Model predictions can decode to negative sizes; MediaPipe drops them.
        if (xmax - xmin < 0 || ymax - ymin < 0 || std::isnan(xmax - xmin) || std::isnan(ymax - ymin)) continue;

        Detection detection;
        detection.score = max_score;
        detection.label_id = class_id;
        detection.box = {xmin, options.flip_vertically ? 1.f - ymax : ymin, xmax - xmin, ymax - ymin};
        for (int k = 0; k < options.num_keypoints; ++k) {
            const int offset = i * options.num_coords + options.keypoint_coord_offset + k * options.num_values_per_keypoint;
            float kx, ky;
            if (format == BoxFormat::kXYWH || format == BoxFormat::kXYXY) {
                kx = raw_boxes[offset];
                ky = raw_boxes[offset + 1];
            } else {
                ky = raw_boxes[offset];
                kx = raw_boxes[offset + 1];
            }
            kx = kx / options.x_scale * anchors[i].w + anchors[i].x_center;
            ky = ky / options.y_scale * anchors[i].h + anchors[i].y_center;
            detection.keypoints.push_back({kx, options.flip_vertically ? 1.f - ky : ky});
        }
        detections.push_back(std::move(detection));
        if (options.max_results > 0 && static_cast<int>(detections.size()) == options.max_results) break;
    }
    return detections;
}

TensorsToDetectionsNode::TensorsToDetectionsNode(TensorsToDetectionsOptions options, std::vector<Anchor> anchors,
                                                 int box_tensor_index, int score_tensor_index)
    : Node("TensorsToDetections"),
      options_(std::move(options)),
      anchors_(std::move(anchors)),
      box_index_(box_tensor_index),
      score_index_(score_tensor_index) {}

void TensorsToDetectionsNode::process() {
    const auto& outputs = tensors.get();
    mx::array boxes = mx::astype(outputs.at(box_index_), mx::float32);
    mx::array scores = mx::astype(outputs.at(score_index_), mx::float32);
    mx::eval(boxes, scores);
    if (boxes.size() != static_cast<size_t>(options_.num_boxes * options_.num_coords) ||
        scores.size() != static_cast<size_t>(options_.num_boxes * options_.num_classes)) {
        throw std::invalid_argument("TensorsToDetections: unexpected tensor sizes");
    }
    detections.send(DecodeDetections(boxes.data<float>(), scores.data<float>(), anchors_, options_));
}

}  // namespace mlxpipe
