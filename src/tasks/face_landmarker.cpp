#include "mlxpipe/tasks/face_landmarker.h"

#include <stdexcept>
#include <tuple>

#include "mlxpipe/calculators/association.h"
#include "mlxpipe/calculators/core.h"
#include "mlxpipe/calculators/detections_to_rects.h"
#include "mlxpipe/calculators/image_to_tensor.h"
#include "mlxpipe/calculators/inference.h"
#include "mlxpipe/calculators/landmarks.h"
#include "mlxpipe/calculators/landmarks_smoothing.h"
#include "mlxpipe/calculators/rect_transformation.h"
#include "mlxpipe/model/model_bundle.h"

namespace mlxpipe {

// face_blendshapes_graph.cc
const std::vector<int> kFaceBlendshapesLandmarkSubset = {
    0,   1,   4,   5,   6,   7,   8,   10,  13,  14,  17,  21,  33,  37,  39,  40,  46,  52,  53,  54,  55,
    58,  61,  63,  65,  66,  67,  70,  78,  80,  81,  82,  84,  87,  88,  91,  93,  95,  103, 105, 107, 109,
    127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163,
    168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283,
    284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324, 332, 334,
    336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382, 384, 385, 386, 387, 388, 389,
    390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477};

const std::vector<std::string> kFaceBlendshapeNames = {
    "_neutral",         "browDownLeft",        "browDownRight",     "browInnerUp",      "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff",           "cheekSquintLeft",   "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight",    "eyeLookDownLeft",     "eyeLookDownRight",  "eyeLookInLeft",    "eyeLookInRight",
    "eyeLookOutLeft",   "eyeLookOutRight",     "eyeLookUpLeft",     "eyeLookUpRight",   "eyeSquintLeft",
    "eyeSquintRight",   "eyeWideLeft",         "eyeWideRight",      "jawForward",       "jawLeft",
    "jawOpen",          "jawRight",            "mouthClose",        "mouthDimpleLeft",  "mouthDimpleRight",
    "mouthFrownLeft",   "mouthFrownRight",     "mouthFunnel",       "mouthLeft",        "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft",   "mouthPressRight",   "mouthPucker",      "mouthRight",
    "mouthRollLower",   "mouthRollUpper",      "mouthShrugLower",   "mouthShrugUpper",  "mouthSmileLeft",
    "mouthSmileRight",  "mouthStretchLeft",    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft",    "noseSneerRight"};

FaceLandmarksDetectorOutputs BuildFaceLandmarksDetectorGraph(Graph& graph, Stream<Image> image,
                                                             Stream<std::vector<NormalizedRect>> face_rects,
                                                             std::shared_ptr<const TfLiteModel> model,
                                                             const FaceLandmarksDetectorGraphOptions& options,
                                                             std::optional<Stream<std::vector<bool>>> keep) {
    const std::vector<int> input = model->input_shape(0);
    const int height = input.at(1), width = input.at(2);
    const int num_landmarks = model->output_shape(0).back() / 3;

    // Tasks preprocessing: stretch the rect to the tensor (no letterbox).
    ImageToTensorOptions to_tensor;
    to_tensor.output_tensor_width = width;
    to_tensor.output_tensor_height = height;
    std::tie(to_tensor.output_tensor_float_range_min, to_tensor.output_tensor_float_range_max) =
        model->input_range({0.0f, 1.0f});
    auto& preprocessing = graph.add<ImageToTensorNode>(to_tensor);
    image >> preprocessing.image;
    face_rects >> preprocessing.rects;

    auto& inference = graph.add<InferenceNode>(model);
    preprocessing.tensor >> inference.tensor;

    // Outputs: 0 landmarks, 1 face flag (logit).
    auto& presence = graph.add<TensorsToFloatsNode>(1, /*sigmoid=*/true);
    inference.tensors >> presence.tensors;
    auto& thresholding = graph.add<ThresholdingNode>(options.min_detection_confidence);
    presence.floats >> thresholding.values;
    Stream<std::vector<bool>> flags = thresholding.flags;
    if (keep) {
        auto& both = graph.add<AndFlagsNode>();
        thresholding.flags >> both.a;
        *keep >> both.b;
        flags = both.output;
    }

    TensorsToLandmarksOptions landmarks_options;
    landmarks_options.num_landmarks = num_landmarks;
    landmarks_options.input_image_width = width;
    landmarks_options.input_image_height = height;
    auto& to_landmarks = graph.add<TensorsToLandmarksNode>(landmarks_options);
    inference.tensors >> to_landmarks.tensors;

    auto& letterbox = graph.add<LandmarkLetterboxRemovalNode>();
    to_landmarks.normalized_landmarks >> letterbox.landmarks;
    preprocessing.letterbox_paddings >> letterbox.paddings;

    auto& projection = graph.add<LandmarkProjectionNode>();
    letterbox.output >> projection.landmarks;
    face_rects >> projection.rects;
    preprocessing.image_size >> projection.image_size;

    // Next-frame rect: rotation from the eye corners (33 -> 263), scaled 1.5.
    auto& to_detection = graph.add<LandmarksToDetectionNode>();
    projection.output >> to_detection.landmarks;
    DetectionsToRectsOptions to_rects;
    to_rects.rotation_vector_start_keypoint_index = 33;
    to_rects.rotation_vector_end_keypoint_index = 263;
    to_rects.rotation_vector_target_angle = 0.0f;
    auto& detection_to_rect = graph.add<DetectionsToRectsNode>(to_rects);
    to_detection.detections >> detection_to_rect.detections;
    preprocessing.image_size >> detection_to_rect.image_size;
    face_rects >> detection_to_rect.source_rects;
    RectTransformationOptions expand;
    expand.scale_x = expand.scale_y = 1.5f;
    expand.square_long = true;
    auto& next_rect = graph.add<RectTransformationNode>(expand);
    detection_to_rect.rects >> next_rect.rects;
    preprocessing.image_size >> next_rect.image_size;

    auto& keep_landmarks = graph.add<FilterByFlagsNode<NormalizedLandmarkList>>();
    projection.output >> keep_landmarks.items;
    flags >> keep_landmarks.flags;
    auto& keep_rects = graph.add<FilterByFlagsNode<NormalizedRect>>();
    next_rect.output >> keep_rects.items;
    flags >> keep_rects.flags;
    auto& keep_scores = graph.add<FilterByFlagsNode<float>>();
    presence.floats >> keep_scores.items;
    flags >> keep_scores.flags;

    return {keep_landmarks.output, keep_rects.output, keep_scores.output};
}

Stream<std::vector<ClassificationList>> BuildFaceBlendshapesGraph(Graph& graph,
                                                                  Stream<std::vector<NormalizedLandmarkList>> landmarks,
                                                                  Stream<ImageSize> image_size,
                                                                  std::shared_ptr<const TfLiteModel> model) {
    auto& to_tensor = graph.add<LandmarksToTensorNode>(kFaceBlendshapesLandmarkSubset);
    landmarks >> to_tensor.landmarks;
    image_size >> to_tensor.image_size;

    auto& inference = graph.add<PerItemInferenceNode>(std::move(model));
    to_tensor.tensors >> inference.items;

    TensorsToClassificationOptions classification;
    classification.label_map = kFaceBlendshapeNames;
    auto& to_classification = graph.add<TensorsToClassificationNode>(classification);
    inference.tensors >> to_classification.tensors;
    return to_classification.classifications;
}

namespace {

template <typename T>
Stream<std::vector<T>> Concat(Graph& graph, Stream<std::vector<T>> a, Stream<std::vector<T>> b) {
    auto& node = graph.add<ConcatenateVectorNode<T>>();
    a >> node.first;
    b >> node.second;
    return node.output;
}

}  // namespace

FaceLandmarkerOutputs BuildFaceLandmarkerGraph(Graph& graph, Stream<Image> image,
                                               std::shared_ptr<const TfLiteModel> detector_model,
                                               std::shared_ptr<const TfLiteModel> landmarks_model,
                                               std::shared_ptr<const TfLiteModel> blendshapes_model,
                                               const FaceLandmarkerGraphOptions& options) {
    const int max_faces = options.face_detector.num_faces;
    auto& image_properties = graph.add<ImagePropertiesNode>();
    image >> image_properties.image;

    FaceLandmarkerOutputs out;
    Stream<std::vector<NormalizedLandmarkList>> landmarks;
    Stream<std::vector<NormalizedRect>> rects_next_frame;

    if (!options.use_stream_mode) {
        FaceDetectorOutputs detector = BuildFaceDetectorGraph(graph, image, detector_model, options.face_detector);
        auto& clip = graph.add<ClipVectorSizeNode<NormalizedRect>>(max_faces);
        detector.expanded_face_rects >> clip.input;
        FaceLandmarksDetectorOutputs faces = BuildFaceLandmarksDetectorGraph(
            graph, image, clip.output, landmarks_model, options.face_landmarks_detector);
        landmarks = faces.landmarks;
        rects_next_frame = faces.face_rects_next_frame;
        out.detections = detector.detections;
    } else {
        auto& previous_rects = graph.loopback<std::vector<NormalizedRect>>(std::vector<NormalizedRect>{});
        auto& has_enough_faces = graph.add<HasMinSizeNode<NormalizedRect>>(max_faces);
        previous_rects.previous >> has_enough_faces.items;
        auto& detector_gate = graph.add<GateNode<Image>>(/*disallow_when_true=*/true);
        image >> detector_gate.input;
        has_enough_faces.result >> detector_gate.condition;
        FaceDetectorOutputs detector =
            BuildFaceDetectorGraph(graph, detector_gate.output, detector_model, options.face_detector);

        auto& association = graph.add<AssociationNormRectNode>(options.min_tracking_confidence, max_faces);
        previous_rects.previous >> association.base_rects;
        detector.expanded_face_rects >> association.rects;

        // Tracked faces run alongside detection; ones replaced by a fresh
        // detection are dropped, and new rects get their own pass.
        FaceLandmarksDetectorOutputs tracked =
            BuildFaceLandmarksDetectorGraph(graph, image, previous_rects.previous, landmarks_model,
                                            options.face_landmarks_detector, Stream<std::vector<bool>>(association.base_kept));
        FaceLandmarksDetectorOutputs found = BuildFaceLandmarksDetectorGraph(
            graph, image, association.new_rects, landmarks_model, options.face_landmarks_detector);

        landmarks = Concat(graph, tracked.landmarks, found.landmarks);
        rects_next_frame = Concat(graph, tracked.face_rects_next_frame, found.face_rects_next_frame);
        rects_next_frame >> previous_rects.next;
        out.detections = detector.detections;

        if (options.smooth_landmarks) {
            OneEuroFilterOptions filter;  // face_landmarks_detector_graph.cc
            filter.min_cutoff = 0.05f;
            filter.beta = 80.0f;
            filter.derivate_cutoff = 1.0f;
            auto& smoothing = graph.add<LandmarksSmoothingNode>(filter);
            landmarks >> smoothing.landmarks;
            rects_next_frame >> smoothing.rects;
            image_properties.size >> smoothing.image_size;
            landmarks = smoothing.output;
        }
    }

    out.landmarks = landmarks;
    out.face_rects_next_frame = rects_next_frame;
    if (blendshapes_model) {
        out.blendshapes = BuildFaceBlendshapesGraph(graph, landmarks, image_properties.size, blendshapes_model);
    }
    return out;
}

// --- Task API ----------------------------------------------------------------

struct FaceLandmarker::Pipeline {
    Graph graph;
    GraphInput<Image> image;
    GraphOutput<std::vector<NormalizedLandmarkList>> landmarks;
    std::optional<GraphOutput<std::vector<ClassificationList>>> blendshapes;
    int64_t last_timestamp_ms = -1;

    Pipeline(const FaceLandmarkerOptions& o, const FaceLandmarker& owner, bool stream_mode) {
        FaceLandmarkerGraphOptions options;
        options.face_detector.num_faces = o.num_faces;
        options.face_detector.min_detection_confidence = o.min_face_detection_confidence;
        options.face_landmarks_detector.min_detection_confidence = o.min_face_presence_confidence;
        options.min_tracking_confidence = o.min_tracking_confidence;
        options.use_stream_mode = stream_mode;
        options.smooth_landmarks = o.smooth_landmarks;
        image = graph.input<Image>("image");
        FaceLandmarkerOutputs out =
            BuildFaceLandmarkerGraph(graph, image, owner.detector_model_, owner.landmarks_model_,
                                     o.output_face_blendshapes ? owner.blendshapes_model_ : nullptr, options);
        landmarks = graph.output(out.landmarks);
        if (out.blendshapes) blendshapes = graph.output(*out.blendshapes);
    }

    FaceLandmarkerResult Run(const Image& frame, int64_t timestamp_ms) {
        image.send(frame);
        graph.run(timestamp_ms * 1000);
        FaceLandmarkerResult result;
        result.face_landmarks = landmarks.value_or({});
        if (blendshapes) result.face_blendshapes = blendshapes->value_or({});
        return result;
    }
};

FaceLandmarker::~FaceLandmarker() = default;

std::unique_ptr<FaceLandmarker> FaceLandmarker::Create(const FaceLandmarkerOptions& options) {
    if (options.model_asset_path.empty()) throw std::invalid_argument("FaceLandmarker: model_asset_path is required");
    ModelBundle bundle = ModelBundle::open(options.model_asset_path);
    std::unique_ptr<FaceLandmarker> landmarker(new FaceLandmarker());
    landmarker->options_ = options;
    landmarker->detector_model_ = TfLiteModel::from_buffer(bundle.file("face_detector.tflite"), "face_detector.tflite");
    landmarker->landmarks_model_ =
        TfLiteModel::from_buffer(bundle.file("face_landmarks_detector.tflite"), "face_landmarks_detector.tflite");
    if (options.output_face_blendshapes) {
        landmarker->blendshapes_model_ =
            TfLiteModel::from_buffer(bundle.file("face_blendshapes.tflite"), "face_blendshapes.tflite");
    }
    return landmarker;
}

FaceLandmarkerResult FaceLandmarker::Detect(const Image& image) {
    if (!image_pipeline_) image_pipeline_ = std::make_unique<Pipeline>(options_, *this, false);
    FaceLandmarkerResult result = image_pipeline_->Run(image, ++image_pipeline_->last_timestamp_ms);
    last_sync_count_ = image_pipeline_->graph.last_eval_count();
    return result;
}

FaceLandmarkerResult FaceLandmarker::DetectForVideo(const Image& image, int64_t timestamp_ms) {
    if (!video_pipeline_) video_pipeline_ = std::make_unique<Pipeline>(options_, *this, true);
    if (timestamp_ms <= video_pipeline_->last_timestamp_ms) {
        throw std::invalid_argument("FaceLandmarker: timestamps must be monotonically increasing");
    }
    video_pipeline_->last_timestamp_ms = timestamp_ms;
    FaceLandmarkerResult result = video_pipeline_->Run(image, timestamp_ms);
    last_sync_count_ = video_pipeline_->graph.last_eval_count();
    return result;
}

}  // namespace mlxpipe
