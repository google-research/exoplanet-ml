package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "beam_prepare_inputs",
    srcs = [
        "beam_prepare_inputs.py",
    ],
    deps = [
        "//astronet/data:preprocess",
        "//beam:utils",
        "//beam/light_curve:light_curve_fns",
        "//light_curve:light_curve_py_pb2",
        "//tf_util:configdict",
    ],
)

py_binary(
    name = "beam_prepare_embedding_inputs",
    srcs = [
        "beam_prepare_embedding_inputs.py",
    ],
    deps = [
        "//astronet/data:preprocess",
        "//beam:utils",
        "//beam/astrowavenet:embedding_fns",
        "//beam/light_curve:light_curve_fns",
        "//light_curve:binning",
        "//tf_util:configdict",
        "//tf_util:example_util",
    ],
)

py_binary(
    name = "beam_reshuffle",
    srcs = [
        "beam_reshuffle.py",
    ],
    deps = ["//tf_util:example_util"],
)
