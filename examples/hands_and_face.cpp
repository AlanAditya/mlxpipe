// Hands and face tracked by ONE graph: both pipelines share the input image,
// and their model invocations are evaluated together, so each frame needs a
// single GPU round trip for palm detection, hand landmarks and the face mesh.
#include "example_util.h"

using namespace mlxpipe;

int main(int argc, char** argv) {
    example::Args args = example::ParseArgs(argc, argv);
    ModelBundle hands = ModelBundle::open(args.models_dir + "/hand_landmarker.task");
    ModelBundle face = ModelBundle::open(args.models_dir + "/face_landmarker.task");

    Graph graph;
    GraphInput<Image> image = graph.input<Image>("image");

    HandLandmarkerGraphOptions hand_options;
    hand_options.hand_detector.num_hands = args.max_items;
    HandLandmarkerOutputs hand_out = BuildHandLandmarkerGraph(
        graph, image, TfLiteModel::from_buffer(hands.file("hand_detector.tflite")),
        TfLiteModel::from_buffer(hands.file("hand_landmarks_detector.tflite")), hand_options);

    FaceLandmarkerGraphOptions face_options;
    FaceLandmarkerOutputs face_out = BuildFaceLandmarkerGraph(
        graph, image, TfLiteModel::from_buffer(face.file("face_detector.tflite")),
        TfLiteModel::from_buffer(face.file("face_landmarks_detector.tflite")), nullptr, face_options);

    auto hand_landmarks = graph.output(hand_out.landmarks);
    auto face_landmarks = graph.output(face_out.landmarks);
    const auto contours = FaceContourConnections();

    return example::RunCamera(args, "mlxpipe hands + face", [&](cv::Mat& frame, int64_t timestamp_ms) {
        image.send(Image(frame));
        graph.run(timestamp_ms * 1000);
        for (const auto& landmarks : face_landmarks.value_or({})) {
            example::DrawLandmarks(frame, landmarks, contours, {0, 220, 255}, {0, 220, 255}, 0);
        }
        for (const auto& landmarks : hand_landmarks.value_or({})) {
            example::DrawLandmarks(frame, landmarks, kHandConnections, {255, 160, 0}, {0, 0, 255}, 4);
        }
        return graph.last_eval_count();
    });
}
