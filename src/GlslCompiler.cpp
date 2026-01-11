#include "GlslCompiler.h"
#include "basicFunctions.h"

#include <stdexcept>
#include <iostream>

std::vector<uint32_t> GlslCompiler::compileGLSLToSpirV(
	const std::string& sourcePath,
	shaderc_shader_kind kind,
	const std::string& filename
)
{
	std::string sourceCode = readFile(sourcePath.c_str());
	
	shaderc::Compiler compiler;
	shaderc::CompileOptions options;

	// 高数 shader 这个是 HLSL 代码
	options.SetSourceLanguage(shaderc_source_language_glsl);

	// vulkan 目标
	options.SetTargetEnvironment(
		shaderc_target_env_vulkan,
		shaderc_env_version_vulkan_1_4
	);


	options.SetAutoBindUniforms(true); // bo、to、so自动映射
	options.SetAutoMapLocations(true); // 自动 location

#ifdef _DEBUG
	options.SetGenerateDebugInfo();
	options.SetOptimizationLevel(shaderc_optimization_level_zero);
#else
	options.SetOptimizationLevel(shaderc_optimization_level_performance);
#endif

	auto result = compiler.CompileGlslToSpv(
		sourceCode,
		kind,
		filename.c_str(),
		options
	);

	if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
		std::string errorMsg = result.GetErrorMessage();
		std::cout <<"compile " << sourcePath << " failed with: " << errorMsg;
		throw std::runtime_error(errorMsg);
	}

	return { result.cbegin(), result.cend() };
}