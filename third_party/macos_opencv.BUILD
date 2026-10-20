load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "opencv_lib",
    hdrs = glob([
        "include/opencv5/opencv2/**/*.hpp",
        "include/opencv5/opencv2/**/*.h",
    ]),
    includes = [
        "include/opencv5",
    ],
    strip_include_prefix = "include/opencv5",
    visibility = ["//visibility:public"],
    linkopts = [
        "-L/opt/homebrew/opt/opencv/lib",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_videoio",
        "-lopencv_highgui",
        "-lopencv_geometry",
        "-lopencv_calib",
        "-lopencv_features",
        "-lopencv_flann",
        "-lopencv_photo",
    ],
)
