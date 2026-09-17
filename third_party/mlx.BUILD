load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "mlx",
    hdrs = glob(["include/mlx/**/*.h"]),
    includes = ["include"],
    linkopts = [
        "-L/usr/local/lib",
        "-lmlx",
        "-framework Metal",
        "-framework Foundation",
        "-framework Accelerate",
    ],
    visibility = ["//visibility:public"],
)
