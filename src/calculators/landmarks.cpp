#include "mlxpipe/calculators/landmarks.h"

#include "mlxpipe/calculators/image_to_tensor.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mlxpipe {

namespace mx = mlx::core;

namespace {

float Activate(TensorsToLandmarksOptions::Activation activation, float value) {
    return activation == TensorsToLandmarksOptions::Activation::kSigmoid ? 1.0f / (1.0f + std::exp(-value)) : value;
}

// Evaluated float32 copy of a model output, viewed as [N, values_per_item].
mx::array ItemsView(const std::vector<Tensor>& tensors, int index) {
    mx::array t = mx::astype(tensors.at(index), mx::float32);
    t = mx::reshape(t, {t.ndim() ? t.shape(0) : 1, -1});
    mx::eval(t);
    return t;
}

}  // namespace

LandmarkList DecodeLandmarks(const float* data, int num_values, const TensorsToLandmarksOptions& options) {
    const int dims = num_values / options.num_landmarks;
    if (dims <= 0) throw std::invalid_argument("TensorsToLandmarks: tensor too small");
    LandmarkList out(options.num_landmarks);
    for (int i = 0; i < options.num_landmarks; ++i) {
        const float* v = data + i * dims;
        Landmark& lm = out[i];
        lm.x = options.flip_horizontally ? options.input_image_width - v[0] : v[0];
        if (dims > 1) lm.y = options.flip_vertically ? options.input_image_height - v[1] : v[1];
        if (dims > 2) lm.z = v[2];
        if (dims > 3) lm.visibility = Activate(options.visibility_activation, v[3]);
        if (dims > 4) lm.presence = Activate(options.presence_activation, v[4]);
    }
    return out;
}

NormalizedLandmarkList NormalizeLandmarks(const LandmarkList& landmarks, const TensorsToLandmarksOptions& options) {
    NormalizedLandmarkList out;
    out.reserve(landmarks.size());
    for (const auto& lm : landmarks) {
        out.push_back({lm.x / options.input_image_width, lm.y / options.input_image_height,
                       lm.z / options.input_image_width / options.normalize_z, lm.visibility, lm.presence});
    }
    return out;
}

void TensorsToLandmarksNode::process() {
    mx::array t = ItemsView(tensors.get(), options_.tensor_index);
    const int n = t.shape(0), values = t.shape(1);
    std::vector<LandmarkList> raw;
    for (int i = 0; i < n; ++i) raw.push_back(DecodeLandmarks(t.data<float>() + i * values, values, options_));
    if (options_.input_image_width > 0 && options_.input_image_height > 0) {
        std::vector<NormalizedLandmarkList> normalized;
        for (const auto& list : raw) normalized.push_back(NormalizeLandmarks(list, options_));
        normalized_landmarks.send(std::move(normalized));
    }
    landmarks.send(std::move(raw));
}

void TensorsToFloatsNode::process() {
    mx::array t = ItemsView(tensors.get(), tensor_index_);
    std::vector<float> out;
    for (int i = 0; i < t.shape(0); ++i) {
        float v = t.data<float>()[i * t.shape(1)];
        out.push_back(sigmoid_ ? 1.0f / (1.0f + std::exp(-v)) : v);
    }
    floats.send(std::move(out));
}

void ThresholdingNode::process() {
    std::vector<bool> out;
    for (float v : values.get()) out.push_back(v >= threshold_);
    flags.send(std::move(out));
}

void TensorsToClassificationNode::process() {
    mx::array t = ItemsView(tensors.get(), options_.tensor_index);
    std::vector<ClassificationList> out;
    for (int i = 0; i < t.shape(0); ++i) {
        const float* scores = t.data<float>() + i * t.shape(1);
        ClassificationList list;
        if (options_.binary_classification) {
            list.push_back({0, scores[0], "", ""});
            list.push_back({1, 1.0f - scores[0], "", ""});
        } else {
            for (int c = 0; c < t.shape(1); ++c) list.push_back({c, scores[c], "", ""});
        }
        for (auto& c : list) {
            if (c.index < static_cast<int>(options_.label_map.size())) {
                c.label = options_.label_map[c.index];
                c.display_name = c.label;
            }
        }
        if (options_.top_k > 0) {
            std::stable_sort(list.begin(), list.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
            if (static_cast<int>(list.size()) > options_.top_k) list.resize(options_.top_k);
        }
        out.push_back(std::move(list));
    }
    classifications.send(std::move(out));
}

NormalizedLandmarkList RemoveLetterbox(const NormalizedLandmarkList& landmarks, const LetterboxPadding& padding) {
    const float left_and_right = padding.left + padding.right;
    const float top_and_bottom = padding.top + padding.bottom;
    NormalizedLandmarkList out = landmarks;
    for (auto& lm : out) {
        lm.x = (lm.x - padding.left) / (1.0f - left_and_right);
        lm.y = (lm.y - padding.top) / (1.0f - top_and_bottom);
        lm.z = lm.z / (1.0f - left_and_right);
    }
    return out;
}

NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const NormalizedRect& rect,
                                        bool ignore_rotation) {
    const float angle = ignore_rotation ? 0.0f : rect.rotation;
    NormalizedLandmarkList out = landmarks;
    for (auto& lm : out) {
        const float x = lm.x - 0.5f;
        const float y = lm.y - 0.5f;
        float new_x = std::cos(angle) * x - std::sin(angle) * y;
        float new_y = std::sin(angle) * x + std::cos(angle) * y;
        lm.x = new_x * rect.width + rect.x_center;
        lm.y = new_y * rect.height + rect.y_center;
        lm.z = lm.z * rect.width;
    }
    return out;
}

NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const Matrix4& m) {
    // Z scale: length of the projected (0,0)-(1,0) segment.
    const float z_scale = std::sqrt(m[0] * m[0] + m[4] * m[4]);
    NormalizedLandmarkList out = landmarks;
    for (auto& lm : out) {
        const float x = lm.x, y = lm.y, z = lm.z;
        lm.x = x * m[0] + y * m[1] + z * m[2] + m[3];
        lm.y = x * m[4] + y * m[5] + z * m[6] + m[7];
        lm.z = z_scale * z;
    }
    return out;
}

LandmarkList ProjectWorldLandmarks(const LandmarkList& landmarks, const NormalizedRect& rect) {
    const float cosa = std::cos(rect.rotation);
    const float sina = std::sin(rect.rotation);
    LandmarkList out = landmarks;
    for (size_t i = 0; i < out.size(); ++i) {
        out[i].x = cosa * landmarks[i].x - sina * landmarks[i].y;
        out[i].y = sina * landmarks[i].x + cosa * landmarks[i].y;
    }
    return out;
}

NormalizedLandmarkList ProjectLandmarks(const NormalizedLandmarkList& landmarks, const NormalizedRect& rect,
                                        const ImageSize& image_size, bool ignore_rotation) {
    RotatedRect roi{rect.x_center * image_size.first, rect.y_center * image_size.second,
                    rect.width * image_size.first, rect.height * image_size.second,
                    ignore_rotation ? 0.0f : rect.rotation};
    return ProjectLandmarks(landmarks, GetRotatedSubRectToRectTransformMatrix(roi, image_size.first, image_size.second));
}

Detection LandmarksToDetection(const NormalizedLandmarkList& landmarks) {
    Detection detection;
    float x_min = std::numeric_limits<float>::max(), y_min = x_min;
    float x_max = std::numeric_limits<float>::lowest(), y_max = x_max;
    for (const auto& lm : landmarks) {
        x_min = std::min(x_min, lm.x);
        x_max = std::max(x_max, lm.x);
        y_min = std::min(y_min, lm.y);
        y_max = std::max(y_max, lm.y);
        detection.keypoints.push_back({lm.x, lm.y});
    }
    detection.box = {x_min, y_min, x_max - x_min, y_max - y_min};
    return detection;
}

void LandmarksToDetectionNode::process() {
    std::vector<Detection> out;
    for (const auto& list : landmarks.get()) out.push_back(LandmarksToDetection(list));
    detections.send(std::move(out));
}

void LandmarksToTensorNode::process() {
    const float w = image_size.get().first, h = image_size.get().second;
    std::vector<Tensor> out;
    for (const auto& list : landmarks.get()) {
        std::vector<int> indices = subset_;
        if (indices.empty()) {
            for (size_t i = 0; i < list.size(); ++i) indices.push_back(static_cast<int>(i));
        }
        std::vector<float> values;
        values.reserve(indices.size() * 2);
        for (int idx : indices) {
            values.push_back(list.at(idx).x * w);
            values.push_back(list.at(idx).y * h);
        }
        out.push_back(mx::array(values.begin(), {1, static_cast<int>(indices.size()), 2}, mx::float32));
    }
    tensors.send(std::move(out));
}

void LandmarkLetterboxRemovalNode::process() {
    std::vector<NormalizedLandmarkList> out;
    for (size_t i = 0; i < landmarks.get().size(); ++i) {
        out.push_back(RemoveLetterbox(landmarks.get()[i], paddings.get().at(i)));
    }
    output.send(std::move(out));
}

void LandmarkProjectionNode::process() {
    std::vector<NormalizedLandmarkList> out;
    for (size_t i = 0; i < landmarks.get().size(); ++i) {
        if (rects.has() && image_size.has()) {
            out.push_back(ProjectLandmarks(landmarks.get()[i], rects.get().at(i), image_size.get()));
        } else if (rects.has()) {
            out.push_back(ProjectLandmarks(landmarks.get()[i], rects.get().at(i)));
        } else if (matrices.has()) {
            out.push_back(ProjectLandmarks(landmarks.get()[i], matrices.get().at(i)));
        } else {
            throw std::logic_error("LandmarkProjection needs rects or matrices");
        }
    }
    output.send(std::move(out));
}

void WorldLandmarkProjectionNode::process() {
    std::vector<LandmarkList> out;
    for (size_t i = 0; i < landmarks.get().size(); ++i) {
        out.push_back(ProjectWorldLandmarks(landmarks.get()[i], rects.get().at(i)));
    }
    output.send(std::move(out));
}

}  // namespace mlxpipe
