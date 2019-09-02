package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "beam_transit_search",
    srcs = [
        "beam_transit_search.py",
    ],
    deps = [
        ":bls_fns",
        ":kepler_id",
        # ":prediction_fns",
        "//light_curve:light_curve_py_pb2",
        "//tf_util:configdict",
        "//beam:utils",
        "//beam/light_curve:light_curve_fns",
        "//box_least_squares:box_least_squares_py_pb2",
        "//experimental/beam/light_curve:transit_fns",
        "//tf_util:config_util",
    ],
)

py_binary(
    name = "beam_astronet_predict",
    srcs = [
        "beam_astronet_predict.py",
    ],
    deps = [
        ":bls_fns",
        ":prediction_fns",
        "//beam:utils",
        "//beam/light_curve:light_curve_fns",
        "//box_least_squares:box_least_squares_py_pb2",
    ],
)

py_library(
    name = "bls_fns",
    srcs = [
        "bls_fns.py",
    ],
    deps = [
        ":bls_scorer",
        "//box_least_squares:box_least_squares_py_pb2",
        "//box_least_squares/python:box_least_squares",
        "//light_curve:light_curve_py_pb2",
    ],
)

py_library(
    name = "bls_scorer",
    srcs = ["bls_scorer.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "kepler_id",
    srcs = [
        "kepler_id.py",
    ],
)

py_library(
    name = "prediction_fns",
    srcs = [
        "prediction_fns.py",
    ],
    deps = [
        "//astronet:models",
        "//astronet/data:preprocess",
        "//tf_util:configdict",
    ],
)
