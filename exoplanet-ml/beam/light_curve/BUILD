package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "light_curve_fns",
    srcs = [
        "light_curve_fns.py",
    ],
    deps = [
        "//light_curve:kepler_io",
        "//light_curve:light_curve_py_pb2",
        "//light_curve:util",
        "//third_party/kepler_spline",
        "//third_party/robust_mean",
    ],
)
