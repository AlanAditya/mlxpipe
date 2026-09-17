load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# Compile our functional library as a static library first
cc_library(
    name = "mlx_vision_lib",
    srcs = [
        "vision.cpp", 
        "tensor_utils.cpp",
        "ssd_anchors.cpp",
        "non_max_suppression.cpp",
        "tensors_to_detections.cpp",
        "one_euro_filter.cpp"
    ],
    hdrs = [
        "vision.h", 
        "tensor_utils.h",
        "ssd_anchors.h",
        "non_max_suppression.h",
        "tensors_to_detections.h",
        "one_euro_filter.h",
        "blaze_palm.h",
        "blaze_hand.h"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@macos_opencv//:opencv_lib",
        "@macos_mlx//:mlx",
    ],
)

cc_binary(
    name = "test_mlx_load",
    srcs = ["test_mlx_load.cpp"],
    deps = [
        "@macos_mlx//:mlx",
    ],
)

# Build the dynamic library (.dylib)
cc_binary(
    name = "libmlx_vision.dylib",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":mlx_vision_lib",
    ],
)
