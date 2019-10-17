package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "beam_prepare_inputs",
    srcs = [
        "beam_prepare_inputs.py",
    ],
    deps = [
        ":process_light_curve",
        "//beam:utils",
    ],
)

py_binary(
    name = "beam_predict",
    srcs = [
        "beam_predict.py",
    ],
    deps = [
        ":prediction_fns",
        ":visualize_fns",
        "//astrowavenet:configurations",
        "//astrowavenet/util:estimator_util",
        "//tf_util:configdict",
    ],
)

py_binary(
    name = "beam_visualize",
    srcs = [
        "beam_visualize.py",
    ],
    deps = [
        ":prediction_fns",
        "//astrowavenet:configurations",
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

py_library(
    name = "visualize_fns",
    srcs = [
        "visualize_fns.py",
    ],
)

py_library(
    name = "prediction_fns",
    srcs = [
        "prediction_fns.py",
    ],
    deps = [
        "//astrowavenet:astrowavenet_model",
        "//astrowavenet/data:kepler_light_curves",
        "//astrowavenet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:example_util",
    ],
)

py_library(
    name = "embedding_fns",
    srcs = [
        "embedding_fns.py",
    ],
    deps = [
        "//astrowavenet:astrowavenet_model",
        "//astrowavenet/data:kepler_light_curves",
        "//tf_util:config_util",
        "//tf_util:configdict",
    ],
)
