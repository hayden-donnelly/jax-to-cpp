#include <iostream>
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tools/hlo_module_loader.h"

int main(int argc, char** argv) 
{
    // Load HloModule from file.
    std::string hlo_filename = "hlo_modules/hlo_comp.txt";
    std::function<void(xla::HloModuleConfig*)> config_modifier_hook =
        [](xla::HloModuleConfig* config) { config->set_seed(42); };
    std::unique_ptr<xla::HloModule> test_module = 
        LoadModuleFromFile(
            hlo_filename, xla::hlo_module_loader_details::Config(),
            "txt", config_modifier_hook
        ).value();
    const xla::HloModuleProto test_module_proto = test_module->ToProto();

    // Run it using JAX C++ Runtime (PJRT).

    // Get a CPU client.
    std::unique_ptr<xla::PjRtClient> client = xla::GetTfrtCpuClient(true).value();

    // Compile XlaComputation to PjRtExecutable.
    xla::XlaComputation xla_computation(test_module_proto);
    xla::CompileOptions compile_options;
    std::unique_ptr<xla::PjRtLoadedExecutable> executable =
        client->Compile(xla_computation, compile_options).value();

    // Input.
    xla::Literal literal_x;
    std::unique_ptr<xla::PjRtBuffer> param_x;
    // Output.
    std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results;
    std::shared_ptr<xla::Literal> result_literal;
    xla::ExecuteOptions execute_options;

    for(int i = 0; i < 1000; i++)
    {
        literal_x = xla::LiteralUtil::CreateR2<float>({{0.1f * (float)i}});
        param_x = client->BufferFromHostLiteral(
            literal_x, client->addressable_devices()[0]
        ).value();

        results = executable->Execute({{param_x.get()}}, execute_options).value();
        result_literal = results[0][0]->ToLiteralSync().value();
        std::cout << "Result " << i << " = " << *result_literal << "\n";
    }
    
    return 0;
}