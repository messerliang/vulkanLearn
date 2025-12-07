
#define VK_EXT_metal_surface
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#if defined(_WIN32) || defined(_WIN64)
#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_EXPOSE_NATIVE_WIN32   // Windows 下暴露 Win32 API
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA   // macOS 下暴露 Cocoa API
#elif defined(__linux__)
#endif

#include <GLFW/glfw3native.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
// glm 的相关 hash 操作
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <chrono>

#include <vulkan/vulkan.h>
#include <optional>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <cstring>
#include <vector>
#include <set>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <unordered_map>

// 一些自定义的变量
#include "defines.h"

// 图片加载 stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// obj 模型加载
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>


// 创建 VkDebugUtilsMessengerEXT，需要调用特定的函数，这个函数需要自己手动找到它
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger);


// debug messenger 同样需要清理
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);


// 用来判断 queue 能力的
struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    
    // 添加一个通用的判断成员函数
    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};


// 用来判断 swap chain 能力的
struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR capabilities;          // min/max image num、min/max image宽高
    std::vector<VkSurfaceFormatKHR> formats;        // surface format(pixel format, color space)
    std::vector<VkPresentModeKHR> presentModes;     // available presentation modes
};


// model view projection struct
struct UniformBufferObject{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};


class VulkanApplication{
public:
    
    VulkanApplication(const std::string& appName="vulkan app",int width=800, int height=600);
    
    void run();
    // 检查配置的 validation layer 是不是有些在硬件里面不支持
    bool checkValidationLayerSupport(bool show=true);
    // 获取需要的 extensions,如果配置了 validation，也需要添加对应的 extension
    std::vector<const char*> getRequiredExtensions(bool showSupport=false);

private: // 静态成员函数
    
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    
    // debug的回调函数，四个参数
    // param0: diagnostic message,判断 message 的严重等级
    // informational message like the creation of a resource
    // message about behavior
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData
    )
    {
        if(messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            
        }
        std::cerr << "validation layers: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
    
public://一些判断函数
    bool isDeviceSuitable(VkPhysicalDevice& device);
    // queue 能力核查
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice& device);
    // 看看显示卡是不是支持特定的功能，如 swap chain（用来绘制图像的）
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    // 查看设备 swap chain 的信息
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    // 获取硬件每个像素都最大采样点
    VkSampleCountFlagBits getMaxUsableSampleCount();
    
    // 具备 swap chain 能力后，还需要确定具体的配置
    // 主要包括下面这些：
    // surface format (color depth)
    // Presentation mode(conditions for "swapping" images to the screen)
    // swap extent (resolution of images in swap chain)
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availiableFormats);
    // 选择一个 present mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    //swap extent，选择分辨率
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    
    // 当窗口大小发生变化的时候，之前 swap chain 中定义的分辨率就不适合了，需要重新创建 swap chain
    void recreateSwapChain();
    
    // 从一组里面找到需要的 format
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
    
    // 选择一个深度 format
    VkFormat findDepthFormat(){
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    /**
     * 给定一个路径，读取文件字节
     */
    static std::vector<char> readFile(const char* filePath);
    
    /**
     * 创建 shafer module
     */
    VkShaderModule createShaderModule(const std::vector<char>& code);
    
    /**
     * 创建 Image 句柄
     * 查询 image 所需要的内存大小与类型
     * 分配并绑定内存到 image
     */
    void createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory);
            
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    
    VkCommandBuffer beginSingleTimeCommands();
    
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
    
    bool hasStencilComponent(VkFormat format){
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
    
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
    
private:
    void initWindow();
    
    
    void initVulkan();
    
    void mainLoop();
    
    void cleanupSwapChain();
    void cleanup();
    
    
    /*
     * 为 image 创建 ImageView
     * @params:
     *  VkImage
     *  VkFormat,描述每个像素占用的字节，
     *      如 VK_FORMAT_R8G8B8A8_UNORM 表示每个像素 4 个字节，范围[0,1]
     *      VK_FORMAT_D32_SFLOAT → 32 位浮点深度缓冲
     *  uint32_t mip map级
     *  VkImageAspectFlags 置顶图像视图的哪一部分，
     *      VK_IMAGE_ASPECT_COLOR_BIT 颜色部份，VK_IMAGE_ASPECT_DEPTH_BIT 深度部份，
     *      VK_IMAGE_ASPECT_STENCIL_BIT 模版部份
     */
    VkImageView createImageView(VkImage image, VkFormat format, uint32_t mipLevels, VkImageAspectFlags aspectFlags);
    
    
    void setupDebugMessenger();  // 启用 debug 功能，配合 validation，可以打印一些信息
    
    void createInstance(bool show=false);       // 启用 extensions，主要包 glfw 需要的、debug、还有不同操作系统相关的
    void createSurface();        // 创建 vulkan 表面，vulkan与窗口系统连接的桥梁，负责呈现到屏幕
    void pickPhysicalDevice(bool show=false);   // 选择一个图形卡，不同的 physical device，支持不同的extensions，这里选择支持 swap chain的物理设备
    void createLogicalDevice();  // 创建 logical device，获取 graphic queue 和 present queue
    void createSwapChain();      // 创建交换链，
    void createImageViews();     // 为每个交换链中的数据创建一个 image views
    void createRenderPass();     // 附件、subpass，render pass
    /**
     * 创建shader对外接口的规范，这里包括一个在 vertex 中的 uniform，还有一个在 fragment 中的纹理采样器
     */
    void createDescriptorSetLayout();
    /**
     * 创建渲染管线
     * 参数0: 编译好的顶点着色器 .spv 文件的路径
     * 参数1: 编译好的片元着色器 .spv 文件的路径
     */
    void createGraphicPipeline(const char* vertSpv, const char* fragSpv);        // 创建渲染管线
    void createCommandPool();            // 创建命令池
    void createColorResources();         // 为 msaa 创建对应的资源，主要包括 image、imageview 还有对应的 memory
    void createDepthResources();         // 深度测试相关的内容
    void createFramebuffers();           // 创建帧缓存，定义一个帧里面有多少个 view port
    void createTextureImage();
    void createTextureImageView();
    void createTextureSampler();
    void loadModel();
    void createVertexBuffer();
    void createIndexBuffer();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffer();
    void createSyncObjects();
    
    
    void drawFrame();

private: // 基本配置
    // 窗口宽度和高度
    std::string m_applicationName;
    uint32_t m_windowWidth  = 800;
    uint32_t m_windowHeight = 600;
    const int MAX_FRAMES_IN_FLIGHT = 4;
    
    // 集成 layer 信息，来帮助debug定位错误
    const std::vector<const char*> m_validationLayers = {
        "VK_LAYER_KHRONOS_validation",
    };
    // 物理设备device需要开启的 extension
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,// 是不是支持 swap chain
    };
    // 定义是否需要进行 debug 信息
    #ifdef NDEBUG
    const bool m_enableValidationLayers = false;
    #else
    const bool m_enableValidationLayers = true;
    #endif
    
    
private: // 相关成员

    GLFWwindow* window;
    
    VkInstance instance;
    
    VkDebugUtilsMessengerEXT debugMessenger;
    
    // widow system integration，主要是配置窗口的
    VkSurfaceKHR surface;
    
    // 物理设备，也就是我们的显卡
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    
    // logic device
    VkDevice m_device;
    
    // 执行渲染的队列 —— 图形队列
    VkQueue graphicsQueue;
    // 将图像显示到屏幕的命令队列 —— 展示队列
    VkQueue presentQueue;
    // 交换链
    VkSwapchainKHR swapChain;
    
    // 需要准备一个 handles 来查看 VkImage
    std::vector<VkImage> swapChainImages;
    // 保存当前的一些信息
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    //要绘制图像，需要准备 VkImageView 这个对象
    std::vector<VkImageView> swapChainImageViews;
    // frame buffer
    std::vector<VkFramebuffer> swapChainFramebuffers;
    
    VkRenderPass renderPass;
    
    // 用来控制 unifoem 参数的
    VkDescriptorSetLayout descriptorSetLayout;
    //uniform values need to be specified during pipeline creation by creating a VkPipelineLayout object.
    VkPipelineLayout pipelineLayout;
    // 最后的渲染管线
    VkPipeline graphicsPipeline;
    
    
    // command pool
    VkCommandPool commandPool;
    
    
    std::vector<Vertex> vertices ;
    std::vector<uint32_t> indices;
    
    // 顶点数据 buffer
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    
    // index buffer
    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    
    // 用于 uniform 数据传输的数组
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    // command buffer
    std::vector<VkCommandBuffer> commandBuffers; // 当 command pool 被释放的时候会自动释放
    
    // 添加几个用于 cpu 和 gpu 同步的变量
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    
    uint32_t currentFrame = 0;
    
    bool framebufferResized = false;
    
    // texture image相关的成员变量
    uint32_t mipLevels;
    VkImage textureImage;
    VkImageView textureImageView;
    VkDeviceMemory textureImageMemory;
    VkSampler textureSampler;

    // depth test，深度测试相关的
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    
    // MSAA 相关的成员变量
    VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;  // 默认情况下每个像素只有 1 个采样点
    // 超采样因为每个像素点都包含有多个采样值，需要的image维度和显示在屏幕上的不一样，所以需要额外的 vkimage 、view、memory等变量来保存
    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;
    
    
};
