#include "mlxpipe/calculators/non_max_suppression.h"

#include <algorithm>
#include <utility>

namespace mlxpipe {

namespace {

using IndexedScores = std::vector<std::pair<int, float>>;

float Area(const RelativeBoundingBox& b) { return b.width * b.height; }

float OverlapSimilarity(NonMaxSuppressionOptions::OverlapType type, const RelativeBoundingBox& a,
                        const RelativeBoundingBox& b) {
    float ix0 = std::max(a.xmin, b.xmin), iy0 = std::max(a.ymin, b.ymin);
    float ix1 = std::min(a.xmin + a.width, b.xmin + b.width), iy1 = std::min(a.ymin + a.height, b.ymin + b.height);
    if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
    float intersection = (ix1 - ix0) * (iy1 - iy0);
    float normalization;
    switch (type) {
        case NonMaxSuppressionOptions::OverlapType::kJaccard:
        case NonMaxSuppressionOptions::OverlapType::kIntersectionOverUnion:
            normalization = Area(a) + Area(b) - intersection;
            break;
        case NonMaxSuppressionOptions::OverlapType::kModifiedJaccard:
            normalization = Area(b);
            break;
    }
    return normalization > 0.0f ? intersection / normalization : 0.0f;
}

std::vector<Detection> DefaultNms(const IndexedScores& indexed_scores, const std::vector<Detection>& detections,
                                  const NonMaxSuppressionOptions& options) {
    std::vector<Detection> output;
    std::vector<const RelativeBoundingBox*> retained;
    for (const auto& [index, score] : indexed_scores) {
        if (options.max_num_detections > 0 && static_cast<int>(output.size()) >= options.max_num_detections) break;
        if (options.min_score_threshold > 0 && score < options.min_score_threshold) break;
        const RelativeBoundingBox& box = detections[index].box;
        bool suppressed = false;
        for (const RelativeBoundingBox* kept : retained) {
            if (OverlapSimilarity(options.overlap_type, *kept, box) > options.min_suppression_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            output.push_back(detections[index]);
            retained.push_back(&detections[index].box);
        }
    }
    return output;
}

std::vector<Detection> WeightedNms(const IndexedScores& indexed_scores, const std::vector<Detection>& detections,
                                   const NonMaxSuppressionOptions& options) {
    std::vector<Detection> output;
    IndexedScores remained = indexed_scores;
    while (!remained.empty()) {
        const size_t original_size = remained.size();
        const Detection& detection = detections[remained[0].first];
        if (options.min_score_threshold > 0 && detection.score < options.min_score_threshold) break;

        IndexedScores candidates, rest;
        for (const auto& indexed : remained) {
            float similarity = OverlapSimilarity(options.overlap_type, detections[indexed.first].box, detection.box);
            (similarity > options.min_suppression_threshold ? candidates : rest).push_back(indexed);
        }

        Detection weighted = detection;
        if (!candidates.empty()) {
            const size_t num_keypoints = detection.keypoints.size();
            std::vector<float> keypoints(num_keypoints * 2, 0.0f);
            float w_xmin = 0, w_ymin = 0, w_xmax = 0, w_ymax = 0, total_score = 0;
            for (const auto& [index, score] : candidates) {
                total_score += score;
                const RelativeBoundingBox& b = detections[index].box;
                w_xmin += b.xmin * score;
                w_ymin += b.ymin * score;
                w_xmax += (b.xmin + b.width) * score;
                w_ymax += (b.ymin + b.height) * score;
                for (size_t k = 0; k < num_keypoints; ++k) {
                    keypoints[k * 2] += detections[index].keypoints[k].x * score;
                    keypoints[k * 2 + 1] += detections[index].keypoints[k].y * score;
                }
            }
            weighted.box.xmin = w_xmin / total_score;
            weighted.box.ymin = w_ymin / total_score;
            weighted.box.width = w_xmax / total_score - weighted.box.xmin;
            weighted.box.height = w_ymax / total_score - weighted.box.ymin;
            for (size_t k = 0; k < num_keypoints; ++k) {
                weighted.keypoints[k] = {keypoints[k * 2] / total_score, keypoints[k * 2 + 1] / total_score};
            }
        }
        output.push_back(std::move(weighted));
        if (original_size == rest.size()) break;
        remained = std::move(rest);
    }
    if (options.max_num_detections > 0 && static_cast<int>(output.size()) > options.max_num_detections) {
        output.resize(options.max_num_detections);
    }
    return output;
}

}  // namespace

std::vector<Detection> NonMaxSuppression(const std::vector<Detection>& detections,
                                         const NonMaxSuppressionOptions& options) {
    IndexedScores indexed_scores;
    for (size_t i = 0; i < detections.size(); ++i) indexed_scores.push_back({static_cast<int>(i), detections[i].score});
    std::stable_sort(indexed_scores.begin(), indexed_scores.end(),
                     [](const auto& a, const auto& b) { return a.second > b.second; });
    return options.algorithm == NonMaxSuppressionOptions::Algorithm::kWeighted
               ? WeightedNms(indexed_scores, detections, options)
               : DefaultNms(indexed_scores, detections, options);
}

}  // namespace mlxpipe
