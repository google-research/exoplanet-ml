package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//devtools/clif/python:clif_build_rule.bzl", "pyclif_proto_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

cc_library(
    name = "bin_by_phase",
    hdrs = ["bin_by_phase.h"],
    deps = ["@com_google_absl//absl/strings"],
)

cc_test(
    name = "bin_by_phase_test",
    srcs = [
        "bin_by_phase_test.cc",
    ],
    deps = [
        ":bin_by_phase",
        ":test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

proto_library(
    name = "box_least_squares_proto",
    srcs = ["box_least_squares.proto"],
    cc_api_version = 2,
)

cc_proto_library(
    name = "box_least_squares_cc_proto",
    deps = [":box_least_squares_proto"],
)

py_proto_library(
    name = "box_least_squares_py_pb2",
    api_version = 2,
    deps = [":box_least_squares_proto"],
)

pyclif_proto_library(
    name = "box_least_squares_pyclif",
    proto_lib = ":box_least_squares_proto",
)

cc_library(
    name = "box_least_squares_impl",
    srcs = ["box_least_squares_impl.cc"],
    hdrs = ["box_least_squares_impl.h"],
    deps = [
        ":box_least_squares_cc_proto",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "box_least_squares_impl_test",
    srcs = [
        "box_least_squares_impl_test.cc",
    ],
    deps = [
        ":box_least_squares_cc_proto",
        ":box_least_squares_impl",
        ":test_util",
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "box_least_squares",
    srcs = ["box_least_squares.cc"],
    hdrs = ["box_least_squares.h"],
    deps = [
        ":bin_by_phase",
        ":box_least_squares_cc_proto",
        ":box_least_squares_impl",
        ":util",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "box_least_squares_test",
    srcs = [
        "box_least_squares_test.cc",
    ],
    deps = [
        ":box_least_squares",
        ":box_least_squares_cc_proto",
        ":test_util",
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "util",
    hdrs = ["util.h"],
)

cc_library(
    name = "test_util",
    hdrs = ["test_util.h"],
    deps = [
        "@com_google_googletest//:gtest",
    ],
)
