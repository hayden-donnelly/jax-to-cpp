cc_binary(
    name = "test",
    srcs = ["src/test.cpp"],
    deps = [
        "@xla//xla:literal",
    ],
)

cc_binary(
    name = "execute_hlo",
    srcs = ["src/execute_hlo.cpp"],
    deps = [
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:tfrt_cpu_pjrt_client",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/tools:hlo_module_loader",
    ],
)