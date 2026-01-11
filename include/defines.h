#ifndef _DEFINES_H_
#define _DEFINES_H_


#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// glm 的相关 hash 操作
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <array>

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    Vertex(
        const glm::vec3& p=glm::vec3(0.0f), 
        const glm::vec3& c=glm::vec3(0.0f), 
        const glm::vec2& uv=glm::vec2(0.0f)
    ):pos(p), color(c), texCoord(uv){}
    
    bool operator==(const Vertex& other) const{
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
    
    static VkVertexInputBindingDescription getBindingDescription(){
        
        VkVertexInputBindingDescription bindingDescription{};
        
        bindingDescription.binding = 0;                 // 这里指定了顶点缓冲区的 id
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        
        return bindingDescription;
    }
    
    static std::array<VkVertexInputAttributeDescription,3> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        
        attributeDescriptions[0].binding = 0;       // 对应哪个顶点缓冲区绑定
        attributeDescriptions[0].location = 0;      // 对应 shader 中 layout(location = x)的x
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);
        
        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        
        return attributeDescriptions;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                   (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                   (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

#endif // !_DEFINES_H_