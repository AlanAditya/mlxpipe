// Mirrors mediapipe/framework/formats/landmark.proto.
#pragma once

#include <vector>

namespace mlxpipe {

// x and y normalized to [0, 1] by image width and height; z uses roughly the
// same scale as x, with smaller values closer to the camera.
struct NormalizedLandmark {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float visibility = 0.0f;
    float presence = 0.0f;
};
using NormalizedLandmarkList = std::vector<NormalizedLandmark>;

// World coordinates in metres.
struct Landmark {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float visibility = 0.0f;
    float presence = 0.0f;
};
using LandmarkList = std::vector<Landmark>;

}  // namespace mlxpipe
