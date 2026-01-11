#pragma once

#include <vector>
#include <shaderc/shaderc.hpp>
#include "basicFunctions.h"

class GlslCompiler
{
public:
	/*
	* 将 hlsl 代码编译为 SPIR-V
	* 参数1：
	*	hlsl 源代码
	* 参数2：
	*	当前代码是哪种类型：vertex shader、fragment shader、 compute shader等等
	*	Vertex			shaderc_vertex_shader
		Fragment		shaderc_fragment_shader
		Compute			shaderc_compute_shader
		Geometry		shaderc_geometry_shader
		Tess Control	shaderc_tess_control_shader
		Tess Eval		shaderc_tess_evaluation_shader
	*/
	std::vector<uint32_t> compileGLSLToSpirV(
		const std::string& source,
		shaderc_shader_kind kind,
		const std::string& filename = "shader.hlsl"

	);
};

