"""A TensorFlow model for generative modeling of light curves."""

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

config_setting(
    name = "tpu_executor_hardware_mode",
    values = {"define": "tpu_hardware=true"},
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":astrowavenet_model",
        ":configurations",
        "//astrowavenet/data:kepler_light_curves",
        "//astrowavenet/data:synthetic_transits",
        "//astrowavenet/util:estimator_util",
        "//tf_util:config_util",
        "//tf_util:configdict",
        "//tf_util:estimator_runner",
    ] + select({
        # Link TPU dependencies if --define=tpu_hardware=true.
        # This allows the trainer to run directly on a TPU host without
        # needing a coordinator.
        ":tpu_executor_hardware_mode": [
            "//learning/brain/google/xla:jellyfish_hardware_device",
            # For Xprof profiling:
            # "//perftools/gputools/profiler:jfprof_lib",
            # "//perftools/gputools/profiler:xprofilez_with_server",
        ],
        "//conditions:default": [],
    }),
)

py_library(
    name = "configurations",
    srcs = ["configurations.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "astrowavenet_model",
    srcs = [
        "astrowavenet_model.py",
    ],
    srcs_version = "PY2AND3",
)

py_test(
    name = "astrowavenet_model_test",
    size = "small",
    srcs = [
        "astrowavenet_model_test.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":astrowavenet_model",
        ":configurations",
        "//tf_util:configdict",
    ],
)
