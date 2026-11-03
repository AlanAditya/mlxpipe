#include "mlxpipe/tasks/face_detector.h"

#include <stdexcept>
#include <tuple>

#include "mlxpipe/calculators/core.h"
#include "mlxpipe/calculators/detection_projection.h"
#include "mlxpipe/calculators/detections_to_rects.h"
#include "mlxpipe/calculators/image_to_tensor.h"
#include "mlxpipe/calculators/inference.h"
#include "mlxpipe/calculators/non_max_suppression.h"
#include "mlxpipe/calculators/rect_transformation.h"
#include "mlxpipe/calculators/ssd_anchors.h"
#include "mlxpipe/calculators/tensors_to_detections.h"
#include "mlxpipe/model/model_bundle.h"

namespace mlxpipe {

FaceDetectorOutputs BuildFaceDetectorGraph(Graph& graph, Stream<Image> image,
                                           std::shared_ptr<const TfLiteModel> model,
                                           const FaceDetectorGraphOptions& options) {
    const std::vector<int> input = model->input_shape(0);
    const int size = input.at(2);
    if (size != 128) throw std::invalid_argument("FaceDetector: only the 128x128 short-range model is supported");

    ImageToTensorOptions to_tensor;
    to_tensor.output_tensor_width = size;
    to_tensor.output_tensor_height = size;
    to_tensor.keep_aspect_ratio = true;
    to_tensor.border_mode = ImageToTensorOptions::BorderMode::kZero;
    std::tie(to_tensor.output_tensor_float_range_min, to_tensor.output_tensor_float_range_max) =
        model->input_range({-1.0f, 1.0f});
    auto& preprocessing = graph.add<ImageToTensorNode>(to_tensor);
    image >> preprocessing.image;

    auto& inference = graph.add<InferenceNode>(model);
    preprocessing.tensor >> inference.tensor;

    // face_detector_graph.cc defaults for the short-range model.
    SsdAnchorsOptions anchors;
    anchors.num_layers = 4;
    anchors.min_scale = 0.1484375f;
    anchors.max_scale = 0.75f;
    anchors.input_size_height = 128;
    anchors.input_size_width = 128;
    anchors.anchor_offset_x = 0.5f;
    anchors.anchor_offset_y = 0.5f;
    anchors.strides = {8, 16, 16, 16};
    anchors.aspect_ratios = {1.0f};
    anchors.fixed_anchor_size = true;
    anchors.interpolated_scale_aspect_ratio = 1.0f;

    TensorsToDetectionsOptions decode;
    decode.num_classes = 1;
    decode.num_boxes = 896;
    decode.num_coords = 16;
    decode.box_coord_offset = 0;
    decode.keypoint_coord_offset = 4;
    decode.num_keypoints = 6;
    decode.num_values_per_keypoint = 2;
    decode.sigmoid_score = true;
    decode.score_clipping_thresh = 100.0f;
    decode.reverse_output_order = true;
    decode.min_score_thresh = options.min_detection_confidence;
    decode.x_scale = decode.y_scale = decode.w_scale = decode.h_scale = 128.0f;
    auto& tensors_to_detections = graph.add<TensorsToDetectionsNode>(decode, GenerateSsdAnchors(anchors));
    inference.tensors >> tensors_to_detections.tensors;

    NonMaxSuppressionOptions nms;
    nms.min_suppression_threshold = options.min_suppression_threshold;
    nms.overlap_type = NonMaxSuppressionOptions::OverlapType::kIntersectionOverUnion;
    nms.algorithm = NonMaxSuppressionOptions::Algorithm::kWeighted;
    auto& suppression = graph.add<NonMaxSuppressionNode>(nms);
    tensors_to_detections.detections >> suppression.detections;

    auto& projection = graph.add<DetectionProjectionNode>();
    suppression.output >> projection.detections;
    preprocessing.matrices >> projection.matrices;

    Stream<std::vector<Detection>> detections = projection.output;
    if (options.num_faces > 0) {
        auto& clip = graph.add<ClipVectorSizeNode<Detection>>(options.num_faces);
        detections >> clip.input;
        detections = clip.output;
    }

    DetectionsToRectsOptions to_rects;
    to_rects.rotation_vector_start_keypoint_index = 0;  // left eye (observer's view)
    to_rects.rotation_vector_end_keypoint_index = 1;    // right eye
    to_rects.rotation_vector_target_angle = 0.0f;
    auto& detections_to_rects = graph.add<DetectionsToRectsNode>(to_rects);
    detections >> detections_to_rects.detections;
    preprocessing.image_size >> detections_to_rects.image_size;

    RectTransformationOptions expand;
    expand.scale_x = expand.scale_y = 1.5f;
    auto& rect_transformation = graph.add<RectTransformationNode>(expand);
    detections_to_rects.rects >> rect_transformation.rects;
    preprocessing.image_size >> rect_transformation.image_size;

    return {detections, detections_to_rects.rects, rect_transformation.output};
}

std::unique_ptr<FaceDetector> FaceDetector::Create(const FaceDetectorOptions& options) {
    std::shared_ptr<TfLiteModel> model;
    const std::string& path = options.model_asset_path;
    if (path.size() > 5 && path.substr(path.size() - 5) == ".task") {
        model = TfLiteModel::from_buffer(ModelBundle::open(path).file("face_detector.tflite"), "face_detector.tflite");
    } else {
        model = TfLiteModel::load(path);
    }
    std::unique_ptr<FaceDetector> detector(new FaceDetector());
    detector->image_ = detector->graph_.input<Image>("image");
    FaceDetectorGraphOptions graph_options;
    graph_options.min_detection_confidence = options.min_detection_confidence;
    graph_options.min_suppression_threshold = options.min_suppression_threshold;
    graph_options.num_faces = options.num_faces;
    FaceDetectorOutputs out = BuildFaceDetectorGraph(detector->graph_, detector->image_, model, graph_options);
    detector->detections_ = detector->graph_.output(out.detections);
    return detector;
}

FaceDetectorResult FaceDetector::Detect(const Image& image) {
    image_.send(image);
    graph_.run(++last_timestamp_ms_ * 1000);
    return {detections_.value_or({})};
}

FaceDetectorResult FaceDetector::DetectForVideo(const Image& image, int64_t timestamp_ms) {
    if (timestamp_ms <= last_timestamp_ms_) {
        throw std::invalid_argument("FaceDetector: timestamps must be monotonically increasing");
    }
    last_timestamp_ms_ = timestamp_ms;
    image_.send(image);
    graph_.run(timestamp_ms * 1000);
    return {detections_.value_or({})};
}

}  // namespace mlxpipe
