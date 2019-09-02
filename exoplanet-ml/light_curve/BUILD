package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "kepler_io",
    srcs = ["kepler_io.py"],
    srcs_version = "PY2AND3",
    deps = [":util"],
)

py_test(
    name = "kepler_io_test",
    size = "small",
    srcs = ["kepler_io_test.py"],
    data = glob([
        "test_data/0114/011442793/kplr*.fits",
    ]),
    srcs_version = "PY2AND3",
    deps = [":kepler_io"],
)

py_library(
    name = "binning",
    srcs = ["binning.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "binning_test",
    size = "small",
    srcs = ["binning_test.py"],
    srcs_version = "PY2AND3",
    deps = [":binning"],
)

py_library(
    name = "periodic_event",
    srcs = ["periodic_event.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "periodic_event_test",
    size = "small",
    srcs = ["periodic_event_test.py"],
    srcs_version = "PY2AND3",
    deps = [":periodic_event"],
)

py_library(
    name = "util",
    srcs = ["util.py"],
    srcs_version = "PY2AND3",
)

py_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":periodic_event",
        ":util",
    ],
)

proto_library(
    name = "light_curve_pb2",
    srcs = ["light_curve.proto"],
    cc_api_version = 2,
)

py_proto_library(
    name = "light_curve_py_pb2",
    api_version = 2,
    deps = [":light_curve_pb2"],
)
