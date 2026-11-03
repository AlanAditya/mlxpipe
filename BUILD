load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library", "cc_test")

# The library: bazel build //:mlxpipe
cc_library(
    name = "mlxpipe",
    srcs = glob(["src/**/*.cpp"]),
    hdrs = glob(["include/**/*.h"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@macos_mlx//:mlx",
        "@macos_opencv//:opencv_lib",
    ],
)

filegroup(
    name = "models",
    srcs = glob(["tools/models/*.task"]),
)

# Webcam examples: bazel run //:hand_tracking -- --max 2
[cc_binary(
    name = name,
    srcs = [
        "examples/%s.cpp" % name,
        "examples/example_util.h",
    ],
    copts = ["-std=c++20"],
    data = [":models"],
    local_defines = ['MLXPIPE_MODELS_DIR=\\"tools/models\\"'],
    deps = [":mlxpipe"],
) for name in [
    "hand_tracking",
    "face_landmarks",
    "hands_and_face",
]]

cc_binary(
    name = "mlxpipe_replay",
    srcs = ["tools/replay/replay.cpp"],
    copts = ["-std=c++20"],
    data = [":models"],
    local_defines = ['MLXPIPE_MODELS_DIR=\\"tools/models\\"'],
    deps = [":mlxpipe"],
)

# Tests: bazel test //:all
[cc_test(
    name = name,
    size = "small",
    srcs = [
        "tests/%s.cpp" % name,
        "tests/test_util.h",
    ],
    copts = ["-std=c++20"],
    data = [":models"] + glob(["tests/data/**"]),
    local_defines = ['MLXPIPE_SOURCE_DIR=\\".\\"'],
    deps = [":mlxpipe"],
) for name in [
    "graph_test",
    "model_golden_test",
    "calculators_test",
    "hand_landmarker_test",
    "face_landmarker_test",
]]
