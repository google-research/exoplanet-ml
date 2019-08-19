load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Abseil C++ libraries.
http_archive(
  name = "com_google_absl",
  urls = ["https://github.com/abseil/abseil-cpp/archive/7c7754fb3ed9ffb57d35fe8658f3ba4d73a31e72.zip"],  # 2019-03-14
  strip_prefix = "abseil-cpp-7c7754fb3ed9ffb57d35fe8658f3ba4d73a31e72",
  sha256 = "71d00d15fe6370220b6685552fb66e5814f4dd2e130f3836fc084c894943753f",
)

# GoogleTest/GoogleMock framework. Used by C++ unit-tests.
http_archive(
  name = "com_google_googletest",
  urls = ["https://github.com/google/googletest/archive/8b6d3f9c4a774bef3081195d422993323b6bb2e0.zip"],  # 2019-03-05
  strip_prefix = "googletest-8b6d3f9c4a774bef3081195d422993323b6bb2e0",
  sha256 = "d21ba93d7f193a9a0ab80b96e8890d520b25704a6fac976fe9da81fffb3392e3",
)

# C++ rules for Bazel.
http_archive(
    name = "rules_cc",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.zip",
        "https://github.com/bazelbuild/rules_cc/archive/b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e.zip",
    ],
    strip_prefix = "rules_cc-b7fe9697c0c76ab2fd431a891dbb9a6a32ed7c3e",
    sha256 = "67412176974bfce3f4cf8bdaff39784a72ed709fc58def599d1f68710b58d68b",
)

###############################################################################
# The following rules are needed for box_least_squares only.
###############################################################################

# bazel-skylib 0.8.0 released 2019.03.20
skylib_version = "0.8.0"
http_archive(
    name = "bazel_skylib",
    type = "tar.gz",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/{}/bazel-skylib.{}.tar.gz".format(skylib_version, skylib_version),
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
)

# Protocol buffers.
http_archive(
    name = "com_google_protobuf",
    urls = ["https://github.com/google/protobuf/archive/v3.9.1.zip"],
    strip_prefix = "protobuf-3.9.1",
    sha256 = "c90d9e13564c0af85fd2912545ee47b57deded6e5a97de80395b6d2d9be64854",
)
load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")
protobuf_deps()
