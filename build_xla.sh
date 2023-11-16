git clone https://github.com/openxla/xla.git
cd xla
bazel build \
    //xla:literal //xla:literal_util //xla:shape_util \
    //xla:status //xla:statusor //xla/pjrt:pjrt_client \
    //xla/pjrt:tfrt_cpu_pjrt_client //xla/service:hlo_proto_cc \
    //xla/tools:hlo_module_loader 