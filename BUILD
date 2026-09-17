load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "test_app",
    srcs = ["main.cpp"],
    deps = [
        "//src:mlx_vision_lib",
        "@macos_opencv//:opencv_lib",
    ],
)
