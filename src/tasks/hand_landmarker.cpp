#include "mlxpipe/tasks/hand_landmarker.h"

#include <cmath>
#include <stdexcept>
#include <tuple>

#include "mlxpipe/calculators/core.h"
#include "mlxpipe/calculators/detection_projection.h"
#include "mlxpipe/calculators/detections_to_rects.h"
#include "mlxpipe/calculators/hand.h"
#include "mlxpipe/calculators/image_to_tensor.h"
#include "mlxpipe/calculators/inference.h"
#include "mlxpipe/calculators/landmarks.h"
#include "mlxpipe/calculators/non_max_suppression.h"
#include "mlxpipe/calculators/rect_transformation.h"
#include "mlxpipe/calculators/ssd_anchors.h"
#include "mlxpipe/calculators/tensors_to_detections.h"
#include "mlxpipe/model/model_bundle.h"

namespace mlxpipe {

HandDetectorOutputs BuildHandDetectorGraph(Graph& graph, Stream<Image> image,
                                           std::shared_ptr<const TfLiteModel> palm_model,
                                           const HandDetectorGraphOptions& options) {
    // hand_detector_graph.cc: letterbox to 192x192 with zero borders, [0, 1].
    ImageToTensorOptions to_tensor;
    to_tensor.output_tensor_width = 192;
    to_tensor.output_tensor_height = 192;
    to_tensor.keep_aspect_ratio = true;
    to_tensor.border_mode = ImageToTensorOptions::BorderMode::kZero;
    std::tie(to_tensor.output_tensor_float_range_min, to_tensor.output_tensor_float_range_max) =
        palm_model->input_range({0.0f, 1.0f});
    auto& preprocessing = graph.add<ImageToTensorNode>(to_tensor);
    image >> preprocessing.image;

    auto& inference = graph.add<InferenceNode>(std::move(palm_model));
    preprocessing.tensor >> inference.tensor;

    SsdAnchorsOptions anchors;
    anchors.num_layers = 4;
    anchors.min_scale = 0.1484375f;
    anchors.max_scale = 0.75f;
    anchors.input_size_height = 192;
    anchors.input_size_width = 192;
    anchors.anchor_offset_x = 0.5f;
    anchors.anchor_offset_y = 0.5f;
    anchors.strides = {8, 16, 16, 16};
    anchors.aspect_ratios = {1.0f};
    anchors.fixed_anchor_size = true;

    TensorsToDetectionsOptions decode;
    decode.num_classes = 1;
    decode.num_boxes = 2016;
    decode.num_coords = 18;
    decode.box_coord_offset = 0;
    decode.keypoint_coord_offset = 4;
    decode.num_keypoints = 7;
    decode.num_values_per_keypoint = 2;
    decode.sigmoid_score = true;
    decode.score_clipping_thresh = 100.0f;
    decode.reverse_output_order = true;
    decode.min_score_thresh = options.min_detection_confidence;
    decode.x_scale = decode.y_scale = decode.w_scale = decode.h_scale = 192.0f;
    auto& tensors_to_detections = graph.add<TensorsToDetectionsNode>(decode, GenerateSsdAnchors(anchors));
    inference.tensors >> tensors_to_detections.tensors;

    NonMaxSuppressionOptions nms;
    nms.min_suppression_threshold = 0.3f;
    nms.overlap_type = NonMaxSuppressionOptions::OverlapType::kIntersectionOverUnion;
    nms.algorithm = NonMaxSuppressionOptions::Algorithm::kWeighted;
    auto& suppression = graph.add<NonMaxSuppressionNode>(nms);
    tensors_to_detections.detections >> suppression.detections;

    auto& projection = graph.add<DetectionProjectionNode>();
    suppression.output >> projection.detections;
    preprocessing.matrices >> projection.matrices;

    DetectionsToRectsOptions to_rects;
    to_rects.rotation_vector_start_keypoint_index = 0;  // wrist
    to_rects.rotation_vector_end_keypoint_index = 2;    // middle finger MCP
    to_rects.rotation_vector_target_angle = M_PI * 0.5f;
    auto& detections_to_rects = graph.add<DetectionsToRectsNode>(to_rects);
    projection.output >> detections_to_rects.detections;
    preprocessing.image_size >> detections_to_rects.image_size;

    RectTransformationOptions expand;
    expand.scale_x = expand.scale_y = 2.6f;
    expand.shift_y = -0.5f;
    expand.square_long = true;
    auto& rect_transformation = graph.add<RectTransformationNode>(expand);
    detections_to_rects.rects >> rect_transformation.rects;
    preprocessing.image_size >> rect_transformation.image_size;

    auto& clip = graph.add<ClipVectorSizeNode<NormalizedRect>>(options.num_hands);
    rect_transformation.output >> clip.input;

    return {projection.output, detections_to_rects.rects, clip.output};
}

HandLandmarksDetectorOutputs BuildHandLandmarksDetectorGraph(Graph& graph, Stream<Image> image,
                                                             Stream<std::vector<NormalizedRect>> hand_rects,
                                                             std::shared_ptr<const TfLiteModel> landmarks_model,
                                                             const HandLandmarksDetectorGraphOptions& options) {
    ImageToTensorOptions to_tensor;
    to_tensor.output_tensor_width = 224;
    to_tensor.output_tensor_height = 224;
    to_tensor.keep_aspect_ratio = true;
    std::tie(to_tensor.output_tensor_float_range_min, to_tensor.output_tensor_float_range_max) =
        landmarks_model->input_range({0.0f, 1.0f});
    auto& preprocessing = graph.add<ImageToTensorNode>(to_tensor);
    image >> preprocessing.image;
    hand_rects >> preprocessing.rects;

    auto& inference = graph.add<InferenceNode>(std::move(landmarks_model));
    preprocessing.tensor >> inference.tensor;

    // Model outputs: 0 landmarks, 1 hand flag, 2 handedness, 3 world landmarks.
    auto& presence = graph.add<TensorsToFloatsNode>(1);
    inference.tensors >> presence.tensors;
    auto& thresholding = graph.add<ThresholdingNode>(options.min_detection_confidence);
    presence.floats >> thresholding.values;

    TensorsToLandmarksOptions landmarks_options;
    landmarks_options.num_landmarks = 21;
    landmarks_options.input_image_width = 224;
    landmarks_options.input_image_height = 224;
    landmarks_options.normalize_z = 0.4f;
    landmarks_options.tensor_index = 0;
    auto& to_landmarks = graph.add<TensorsToLandmarksNode>(landmarks_options);
    inference.tensors >> to_landmarks.tensors;

    TensorsToLandmarksOptions world_options;
    world_options.num_landmarks = 21;
    world_options.tensor_index = 3;
    auto& to_world = graph.add<TensorsToLandmarksNode>(world_options);
    inference.tensors >> to_world.tensors;

    TensorsToClassificationOptions handedness_options;
    handedness_options.binary_classification = true;
    handedness_options.label_map = {"Left", "Right"};
    handedness_options.top_k = 1;
    handedness_options.tensor_index = 2;
    auto& to_handedness = graph.add<TensorsToClassificationNode>(handedness_options);
    inference.tensors >> to_handedness.tensors;

    auto& letterbox = graph.add<LandmarkLetterboxRemovalNode>();
    to_landmarks.normalized_landmarks >> letterbox.landmarks;
    preprocessing.letterbox_paddings >> letterbox.paddings;

    auto& projection = graph.add<LandmarkProjectionNode>();
    letterbox.output >> projection.landmarks;
    hand_rects >> projection.rects;

    auto& world_projection = graph.add<WorldLandmarkProjectionNode>();
    to_world.landmarks >> world_projection.landmarks;
    hand_rects >> world_projection.rects;

    auto& to_rect = graph.add<HandLandmarksToRectNode>();
    projection.output >> to_rect.landmarks;
    preprocessing.image_size >> to_rect.image_size;
    hand_rects >> to_rect.source_rects;

    RectTransformationOptions expand;
    expand.scale_x = expand.scale_y = 2.0f;
    expand.shift_y = -0.1f;
    expand.square_long = true;
    auto& next_rect = graph.add<RectTransformationNode>(expand);
    to_rect.rects >> next_rect.rects;
    preprocessing.image_size >> next_rect.image_size;

    // Drop hands below the presence threshold (AllowIf on each output).
    auto& keep_landmarks = graph.add<FilterByFlagsNode<NormalizedLandmarkList>>();
    projection.output >> keep_landmarks.items;
    thresholding.flags >> keep_landmarks.flags;
    auto& keep_world = graph.add<FilterByFlagsNode<LandmarkList>>();
    world_projection.output >> keep_world.items;
    thresholding.flags >> keep_world.flags;
    auto& keep_handedness = graph.add<FilterByFlagsNode<ClassificationList>>();
    to_handedness.classifications >> keep_handedness.items;
    thresholding.flags >> keep_handedness.flags;
    auto& keep_rects = graph.add<FilterByFlagsNode<NormalizedRect>>();
    next_rect.output >> keep_rects.items;
    thresholding.flags >> keep_rects.flags;
    auto& keep_scores = graph.add<FilterByFlagsNode<float>>();
    presence.floats >> keep_scores.items;
    thresholding.flags >> keep_scores.flags;

    return {keep_landmarks.output, keep_world.output, keep_handedness.output, keep_rects.output, keep_scores.output};
}

namespace {

// Concatenates two optional vector streams.
template <typename T>
Stream<std::vector<T>> Concat(Graph& graph, Stream<std::vector<T>> a, Stream<std::vector<T>> b) {
    auto& node = graph.add<ConcatenateVectorNode<T>>();
    a >> node.first;
    b >> node.second;
    return node.output;
}

}  // namespace

HandLandmarkerOutputs BuildHandLandmarkerGraph(Graph& graph, Stream<Image> image,
                                               std::shared_ptr<const TfLiteModel> palm_model,
                                               std::shared_ptr<const TfLiteModel> landmarks_model,
                                               const HandLandmarkerGraphOptions& options) {
    const int max_hands = options.hand_detector.num_hands;
    auto& image_properties = graph.add<ImagePropertiesNode>();
    image >> image_properties.image;

    if (!options.use_stream_mode) {
        HandDetectorOutputs detector = BuildHandDetectorGraph(graph, image, palm_model, options.hand_detector);
        HandLandmarksDetectorOutputs hands = BuildHandLandmarksDetectorGraph(
            graph, image, detector.hand_rects, landmarks_model, options.hand_landmarks_detector);
        auto& dedup = graph.add<HandLandmarksDeduplicationNode>();
        hands.landmarks >> dedup.landmarks;
        hands.world_landmarks >> dedup.world_landmarks;
        hands.hand_rects_next_frame >> dedup.rois;
        hands.handedness >> dedup.classifications;
        image_properties.size >> dedup.image_size;
        return {dedup.landmarks_out, dedup.world_landmarks_out, dedup.classifications_out, dedup.rois_out,
                detector.palm_detections};
    }

    // Hand rects predicted from the previous frame's landmarks.
    auto& previous_rects = graph.loopback<std::vector<NormalizedRect>>(std::vector<NormalizedRect>{});

    // Skip palm detection when enough hands are already tracked.
    auto& has_enough_hands = graph.add<HasMinSizeNode<NormalizedRect>>(max_hands);
    previous_rects.previous >> has_enough_hands.items;
    auto& detector_gate = graph.add<GateNode<Image>>(/*disallow_when_true=*/true);
    image >> detector_gate.input;
    has_enough_hands.result >> detector_gate.condition;
    HandDetectorOutputs detector = BuildHandDetectorGraph(graph, detector_gate.output, palm_model, options.hand_detector);

    // Keep tracked hands; add palm-derived rects that don't overlap them.
    auto& association = graph.add<HandAssociationNode>(options.min_tracking_confidence, max_hands);
    previous_rects.previous >> association.base_rects;
    detector.hand_rects >> association.rects;

    // MediaPipe runs landmarks on the associated rects. Association keeps
    // every tracked rect, so tracked hands are processed straight from the
    // loopback (alongside palm detection) and only new rects run afterwards.
    HandLandmarksDetectorOutputs tracked = BuildHandLandmarksDetectorGraph(
        graph, image, previous_rects.previous, landmarks_model, options.hand_landmarks_detector);
    HandLandmarksDetectorOutputs found = BuildHandLandmarksDetectorGraph(
        graph, image, association.new_rects, landmarks_model, options.hand_landmarks_detector);

    auto& dedup = graph.add<HandLandmarksDeduplicationNode>();
    Concat(graph, tracked.landmarks, found.landmarks) >> dedup.landmarks;
    Concat(graph, tracked.world_landmarks, found.world_landmarks) >> dedup.world_landmarks;
    Concat(graph, tracked.hand_rects_next_frame, found.hand_rects_next_frame) >> dedup.rois;
    Concat(graph, tracked.handedness, found.handedness) >> dedup.classifications;
    image_properties.size >> dedup.image_size;

    dedup.rois_out >> previous_rects.next;
    return {dedup.landmarks_out, dedup.world_landmarks_out, dedup.classifications_out, dedup.rois_out,
            detector.palm_detections};
}

// --- Task API ----------------------------------------------------------------

struct HandLandmarker::Pipeline {
    Graph graph;
    GraphInput<Image> image;
    GraphOutput<std::vector<NormalizedLandmarkList>> landmarks;
    GraphOutput<std::vector<LandmarkList>> world_landmarks;
    GraphOutput<std::vector<ClassificationList>> handedness;
    int64_t last_timestamp_ms = -1;

    Pipeline(const HandLandmarkerOptions& o, std::shared_ptr<const TfLiteModel> palm,
             std::shared_ptr<const TfLiteModel> hand, bool stream_mode) {
        HandLandmarkerGraphOptions options;
        options.hand_detector.min_detection_confidence = o.min_hand_detection_confidence;
        options.hand_detector.num_hands = o.num_hands;
        options.hand_landmarks_detector.min_detection_confidence = o.min_hand_presence_confidence;
        options.min_tracking_confidence = o.min_tracking_confidence;
        options.use_stream_mode = stream_mode;
        image = graph.input<Image>("image");
        HandLandmarkerOutputs out = BuildHandLandmarkerGraph(graph, image, palm, hand, options);
        landmarks = graph.output(out.landmarks);
        world_landmarks = graph.output(out.world_landmarks);
        handedness = graph.output(out.handedness);
    }

    HandLandmarkerResult Run(const Image& frame, int64_t timestamp_ms) {
        image.send(frame);
        graph.run(timestamp_ms * 1000);
        HandLandmarkerResult result;
        result.hand_landmarks = landmarks.value_or({});
        result.hand_world_landmarks = world_landmarks.value_or({});
        result.handedness = handedness.value_or({});
        return result;
    }
};

HandLandmarker::HandLandmarker() = default;
HandLandmarker::~HandLandmarker() = default;

std::unique_ptr<HandLandmarker> HandLandmarker::Create(const HandLandmarkerOptions& options) {
    std::unique_ptr<HandLandmarker> landmarker(new HandLandmarker());
    landmarker->options_ = options;
    if (!options.model_asset_path.empty()) {
        ModelBundle bundle = ModelBundle::open(options.model_asset_path);
        landmarker->palm_model_ = TfLiteModel::from_buffer(bundle.file("hand_detector.tflite"), "hand_detector.tflite");
        landmarker->landmarks_model_ =
            TfLiteModel::from_buffer(bundle.file("hand_landmarks_detector.tflite"), "hand_landmarks_detector.tflite");
    } else if (!options.hand_detector_model_path.empty() && !options.hand_landmarks_model_path.empty()) {
        landmarker->palm_model_ = TfLiteModel::load(options.hand_detector_model_path);
        landmarker->landmarks_model_ = TfLiteModel::load(options.hand_landmarks_model_path);
    } else {
        throw std::invalid_argument("HandLandmarker: set model_asset_path or both model paths");
    }
    return landmarker;
}

HandLandmarkerResult HandLandmarker::Detect(const Image& image) {
    if (!image_pipeline_) image_pipeline_ = std::make_unique<Pipeline>(options_, palm_model_, landmarks_model_, false);
    HandLandmarkerResult result = image_pipeline_->Run(image, 0);
    last_sync_count_ = image_pipeline_->graph.last_eval_count();
    return result;
}

HandLandmarkerResult HandLandmarker::DetectForVideo(const Image& image, int64_t timestamp_ms) {
    if (!video_pipeline_) video_pipeline_ = std::make_unique<Pipeline>(options_, palm_model_, landmarks_model_, true);
    if (timestamp_ms <= video_pipeline_->last_timestamp_ms) {
        throw std::invalid_argument("HandLandmarker: timestamps must be monotonically increasing");
    }
    video_pipeline_->last_timestamp_ms = timestamp_ms;
    HandLandmarkerResult result = video_pipeline_->Run(image, timestamp_ms);
    last_sync_count_ = video_pipeline_->graph.last_eval_count();
    return result;
}

int HandLandmarker::last_gpu_sync_count() const { return last_sync_count_; }

}  // namespace mlxpipe
