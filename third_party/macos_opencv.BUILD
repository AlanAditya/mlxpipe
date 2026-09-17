load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "opencv_lib",
    hdrs = glob([
        "include/opencv4/opencv2/**/*.hpp",
        "include/opencv4/opencv2/**/*.h",
    ]),
    includes = [
        "include/opencv4",
    ],
    strip_include_prefix = "include/opencv4",
    visibility = ["//visibility:public"],
    linkopts = [
        "-L/opt/homebrew/opt/opencv/lib",
        "-lopencv_core",
        "-lopencv_imgproc",
        "-lopencv_imgcodecs",
        "-lopencv_videoio",
        "-lopencv_highgui",
        "-lopencv_calib3d",
        "-lopencv_features2d",
        "-lopencv_flann",
        "-lopencv_photo",
    ],
)
