#pragma once

#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

class Camera
{
public:

private:
	// 窗口对象
	GLFWwindow* m_window = nullptr;
	// 摄像机移动速度
	float m_speed = 3.0f;

};

