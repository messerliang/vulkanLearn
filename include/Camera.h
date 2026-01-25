#ifndef _CAMERA_H_
#define _CAMERA_H_


#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Camera
{
public:
	
	Camera(GLFWwindow* window=nullptr) :m_window(window) {

	}
	~Camera(){
		m_window = nullptr;
	}

	// 设置 window
	void setWindow(GLFWwindow* window) {
		m_window = window;
	}

	glm::vec3 getPosition() const {
		return m_position;
	}

	float getFov() const {
		return m_fov;
	}

	// 根据鼠标滚轮，更新 m_fov
	void updateFov(double yOffset) {
		m_fov -= static_cast<float>(yOffset);
		m_fov = glm::clamp(m_fov, 0.1f, 160.0f);
		std::cout << "fov: " << m_fov << std::endl;
	}

	// 更新按键事件
	void updateKeyEvent(float deltaTime) {
		// 更新位置移动
		updatePosition(deltaTime);
	}

	// 根据键盘按键，更新摄像机位置
	void updatePosition(float deltaTime) {
		// 移动
		glm::vec3 dir{ 0.0f };
		if (glfwGetKey(m_window, GLFW_KEY_D) == GLFW_PRESS) {
			dir.x -= 1.0f;
		}
		if (glfwGetKey(m_window, GLFW_KEY_A) == GLFW_PRESS) {
			dir.x += 1.0f;
		}
		if (glfwGetKey(m_window, GLFW_KEY_Q) == GLFW_PRESS) {
			dir.y -= 1.0f;
		}
		if (glfwGetKey(m_window, GLFW_KEY_E) == GLFW_PRESS) {
			dir.y += 1.0f;
		}
		if (glfwGetKey(m_window, GLFW_KEY_W) == GLFW_PRESS) {
			dir.z -= 1.0f;
		}
		if (glfwGetKey(m_window, GLFW_KEY_S) == GLFW_PRESS) {
			dir.z += 1.0f;
		}

		if (glm::length(dir) > 0) {
			m_position += glm::normalize(dir) * m_speed * deltaTime;
			std::cout << "camera position: " << 1.0f / deltaTime << ": " << m_position.x << ", " << m_position.y << ", " << m_position.z << std::endl;
		}
	}

	glm::vec3 getDirection()const {
		return m_direction;
	}

	glm::vec3 getTargetPoint()const {
		return m_position + glm::normalize(m_direction);
	}

private:
	
	// 窗口对象
	GLFWwindow* m_window = nullptr;
	// 摄像机移动速度
	float m_speed = 200.0f;
	// 摄像机位置
	glm::vec3 m_position{ 3.8, 110, 261 };
	// 摄像机看的方向，默认看向 -Z 的方向
	glm::vec3 m_direction{0.0f, 0.0f, -1.0f};
	// fov
	float m_fov = 45.0f;

};

#endif // !_CAMERA_H_