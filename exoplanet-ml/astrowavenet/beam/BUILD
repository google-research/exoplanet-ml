package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "prepare_astrowavenet_inputs",
    srcs = [
        "prepare_astrowavenet_inputs.py",
    ],
    deps = [
        ":process_light_curve",
        ":utils",
    ],
)

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
)

py_library(
    name = "process_light_curve",
    srcs = [
        "process_light_curve.py",
    ],
    deps = [
        "//light_curve:kepler_io",
        "//light_curve:util",
        "//tf_util:example_util",
    ],
)
