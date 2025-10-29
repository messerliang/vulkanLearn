//
//  test01_helloTriangle.cpp
//  vulkanTesting
//
//  Created by Zhupei Li on 2025/9/27.
//

#if 1


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
#include <glm/glm.hpp>
#include <sstream>

// 图片加载 stb_image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// 窗口常量

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 4;
// 集成 layer 信息，来帮助debug定位错误
const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

// 物理设备device需要开启的 extension
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,// 是不是支持 swap chain
};
// 定义是否需要进行 debug 信息
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif






// 创建 VkDebugUtilsMessengerEXT，需要调用特定的函数，这个函数需要自己手动找到它
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    } else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

// debug messenger 同样需要清理
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// 用来判断 queuq 能力的
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

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    
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
// model view projection struct
struct UniformBufferObject{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};
// 顶点和颜色数据
const std::vector<Vertex> vertices = {
//    {{-0.5, -0.5}, {1.0f, 0.0f, 0.0f},{0.0f, 1.0f}},
//    {{ 0.5, -0.5}, {0.0f, 1.0f, 1.0f},{1.0f, 1.0f}},
//    {{ 0.5,  0.5}, {0.0f, 1.0f, 0.0f},{1.0f, 0.0f}},
//    {{-0.5,  0.5}, {0.0f, 0.0f, 1.0f},{0.0f, 0.0f}},
    {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},

    {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
    {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}
};


// index buffer
const std::vector<uint16_t> indices = {
    0,1,2,
    0,2,3,
    4,5,6,
    6,7,4,
};


class HelloTriangleApplication
{
    
public:
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
    
private:
    GLFWwindow* window;
    
    VkInstance instance;
    
    VkDebugUtilsMessengerEXT debugMessenger;
    
    // widow system integration，主要是配置窗口的
    VkSurfaceKHR surface;
    
    // 物理设备，也就是我们的显卡
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    
    // logic device
    VkDevice device;
    
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
    VkImage textureImage;
    VkImageView textureImageView;
    VkDeviceMemory textureImageMemory;
    VkSampler textureSampler;

    // depth test，深度测试相关的
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    
private:
    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        
        
        // 创建窗口
        window = glfwCreateWindow(WIDTH, HEIGHT, "vulkan", nullptr, nullptr);
        
        // glfw 可以设置一个用户自定义的指针，所以，我们可以把对象的this指针保存下来
        glfwSetWindowUserPointer(window, this);
        
        // 窗口大小发生变化时候到回调函数
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    }
    
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height){
        auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }
    
    void initVulkan()
    {
        
        createInstance();       // 启用 extensions，主要包 glfw 需要的、debug、还有不同操作系统相关的
        setupDebugMessenger();  // 启用 debug 功能，配合 validation，可以打印一些信息
        createSurface();        // 创建 vulkan 表面，vulkan与窗口系统连接的桥梁，负责呈现到屏幕
        pickPhysicalDevice();   // 选择一个图形卡，不同的 physical device，支持不同的extensions，这里选择支持 swap chain的物理设备
        createLogicalDevice();  // 创建 logical device，获取 graphic queue 和 present queue
        createSwapChain();      // 创建交换链，
        createImageViews();     // 为每个交换链中的数据创建一个 image views
        createRenderPass();     // 附件、subpass，render pass
        createDescriptorSetLayout();    // 创建描述符集布局，定义了着色器如何访问资源（uniform 缓冲区、纹理等）的接口“规范”
        createGraphicPipeline();        // 创建渲染管线
        createCommandPool();            // 创建命令池
        createDepthResources();         // 深度测试相关的内容
        createFramebuffers();           // 创建帧缓存，定义一个帧里面有多少个 view port
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffer();
        createSyncObjects();
    }
    void mainLoop()
    {
        
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device);
    }
    
    
    
    void cleanupSwapChain()
    {
        
        // 释放 framebuffer
        for(auto framebuffer:swapChainFramebuffers){
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        // 释放 image view
        for (auto imageView : swapChainImageViews) {
           vkDestroyImageView(device, imageView, nullptr);
        }
        // 释放 swap chain
        vkDestroySwapchainKHR(device, swapChain, nullptr);
        
    }
    
    void cleanup()
    {
        
        cleanupSwapChain();
        
        // 释放纹理相关的内存和变量
        vkDestroyImage(device, textureImage, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);
        // 释放深度测试相关的资源
        vkDestroyImage(device, depthImage, nullptr);
        vkDestroyImageView(device, depthImageView, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);
        
        // 释放 VkSampler
        vkDestroySampler(device, textureSampler, nullptr);
        
        // 释放 uniform map
        for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; ++i){
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        //清除 layout descriptorr
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        
        // 释放 vertex buffer
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        
        // 释放 index buffer
        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        
        
        // 释放用于同步的 semaphore 和 fence
        for(int i=0; i<MAX_FRAMES_IN_FLIGHT; ++i){
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        
        // 释放 command pool
        vkDestroyCommandPool(device, commandPool, nullptr);
        
        
        // 释放 graphics pipeline
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        
        // 释放 pipeline layout
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        // 释放 renderPass
        vkDestroyRenderPass(device, renderPass, nullptr);
        
        
        // 释放 logical device
        vkDestroyDevice(device, nullptr);
        
        if(enableValidationLayers)
        {
            DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
        }
        
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    
    void createInstance(bool show=false)
    {
        // app名字、版本信息
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "hello triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "NO ENGINE";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_4;
        
        // instance
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        
        // layer 配置，看看是不是配置的layer，当前硬件存在不支持的情况
        if(enableValidationLayers && !checkValidationLayerSupport())
        {
            throw std::runtime_error("validation layer requisted, but not supported\n");
        }
        
        if(enableValidationLayers)
        {
            createInfo.enabledLayerCount = (uint32_t)validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }else
        {
            createInfo.enabledLayerCount = 0;
        }
        
        // extension 配置，主要包括 glfw 需要的 extension、debug需要的，不同系统需要的 extensions
        std::vector<const char*> requiredExtensions = getRequiredExtensions();
        
        createInfo.enabledExtensionCount = (uint32_t)requiredExtensions.size();
        createInfo.ppEnabledExtensionNames = requiredExtensions.data();

        
        if(show){
            std::cout << "current extensions num: " << requiredExtensions.size() << std::endl;
        }
        
#ifdef __APPLE__
        createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR; // macos需要的，启用端口性枚举
#endif
        
        if(vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("fail to create instance!");
        }
        
    }
    
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = debugCallback;
    }
    
    void setupDebugMessenger()
    {
        if(!enableValidationLayers)return;
        
        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateDebugMessengerCreateInfo(createInfo);
        
        if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to setup debug messenger!");
        }
        
    }
    
    void createSurface()
    {
        
        if( glfwCreateWindowSurface(instance, window, nullptr, &surface)!= VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface with the error");
        }
    }
    
    
    // 初始化完成后，需要选择一个图形卡来进行绘制 —— 从所有的 physical card 里面，选择一张来进行渲染
    void pickPhysicalDevice(bool show=false)
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if(show){
            std::cout << "num of graphic card: " << deviceCount << std::endl;
            
        }
        
        if(0 == deviceCount)
        {
            throw std::runtime_error("failed to find gpu with vulkan support");
        }
        
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
        
        // 从这些卡里面选择一张来进行计算
        for(auto& device:devices)
        {
            if(isDeviceSuitable(device))
            {
                physicalDevice = device;
                break;
            }
        }
        
        if(VK_NULL_HANDLE == physicalDevice)
        {
            throw std::runtime_error("failed to find suitable GPU");
        }
        
        
    }
    
    
    // 创建 logical device，还需要进行 queue 的创建
    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        
        
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamalies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        
        // 需要一个 float priority [0,1] 来确定优先级，这里主要建立的是 graphic queue 和 present queue
        float queuePriority = 1.0f;
        for(uint32_t queueFamily:uniqueQueueFamalies)
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }
        
        
        // deviceFeatures暂时就不进行配置了
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        
        // logical device 的创建信息
        VkDeviceCreateInfo createInfo{};
        createInfo.pEnabledFeatures = &deviceFeatures;
        
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        
        
        
        // logical device 的validationLayer在最新的 vulkan 版本中已经被废弃了，这里只是为了和老版本兼容，继续配置
        if(enableValidationLayers)
        {
            createInfo.enabledLayerCount = (uint32_t)(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else{
            createInfo.enabledLayerCount = 0;
        }
        
        // 在 macos 平台下，需要启用 VK_KHR_portability_subset extension
        std::vector<const char*> extensions = {
#ifdef __APPLE__
            "VK_KHR_portability_subset",
#endif
        };
        // 然后再加上我们前面全局变量那里配置的 requiredExtensions
        for(auto& extension:deviceExtensions)
        {
            extensions.push_back(extension);
        }
        createInfo.enabledExtensionCount = (uint32_t)(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
        
        // 暂时不需要 extension 了，这里可以进行设备的初始化了
        if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create logical device!\n");
        }
        
        // 读取 graphics queue
        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        // 读取 present queue
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }
    
    
    void createSwapChain(){
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);       // 查询物理设备的显示信息
        
        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);   // 选择颜色格式
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);    // 呈现模式
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);                    // 分辨率，通常是窗口大小
        
        // 确定 swap chain 里面有多少数量的 images
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        
        // 也不要超过支持的最大值
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;                                // 除来 VR 外，基本都是1
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;    // 设置为颜色附件，用于最终颜色输出，也有其他用于后处理的
        
        // 判断绘图队列和显示队列是不是相同——不同：并发模式，CONCURRENT；相同：独占模式，EXCLUSIVE，性能更高
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }
        
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        
        
        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }
        
        // swapChain 创建完成后，可以去获取一些 image 中的内容了，同样，我们需要先获取数量
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        
        std::cout << "num of image in swap chain : " << imageCount << std::endl;
        
        // 保存一下当前的图像格式、分辨率
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
        std::cout <<"width and height: "<< swapChainExtent.width <<" "<< swapChainExtent.height << std::endl;
    }
    
    
    // 当窗口大小发生变化的时候，之前 swap chain 中定义的分辨率就不适合了，需要重新创建 swap chain
    void recreateSwapChain(){
        // 先处理一种特殊的窗口大小变化 —— 最小化
        int width = 0;
        int height = 0;
//        glfwGetWindowSize(window, &width, &height);
        while(width == 0 || height == 0)
        {
            glfwGetWindowSize(window, &width, &height);
            glfwWaitEvents();
        }
        
        // 等待 gpu 空闲
        vkDeviceWaitIdle(device);
        
        cleanupSwapChain();
        
        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
        
    }
    
    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags=VK_IMAGE_ASPECT_COLOR_BIT) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image view!");
        }

        return imageView;
    }
    
    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size());
        // 遍历 ImageViews
        for(size_t i = 0; i < swapChainImages.size(); ++i){
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }
    }
    
    
    void createRenderPass(){
        // 附件管理，主要包括颜色附件、引用附件
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;       // what to do before rendering
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;     // what to do after rendering,这里配置的是渲染后的contents会存在内存里面，可以后面再读取
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; // 配置image to be ready for presentation using swap chain after rendering
        
        // 下面是子通道定义，颜色引用附件
        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        
        // 深度附件
        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        
        // 深度引用附件
        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        
        // subpass
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;
        
        VkSubpassDependency dependency{};
       dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
       dependency.dstSubpass = 0;
       dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
       dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
       dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
       dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        
        // 创建渲染通道
        std::array<VkAttachmentDescription,2> attachments = {colorAttachment, depthAttachment};
        
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = attachments.size();                 // 颜色附件，深度附件或其他模版附件的数量
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;
        

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
    }
    
    // 创建描述符集布局，定义了着色器如何访问资源（uniform 缓冲区、纹理等）的接口“规范”
    void createDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;                               // 着色器中的绑定点，对应 layout(binding=0)
        uboLayoutBinding.descriptorCount = 1;                       // 资源数量
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;  // 资源类型，这里表示是 uniform 缓冲区
        uboLayoutBinding.pImmutableSamplers = nullptr; // 图像采样相关的内容     // 采样器，纹理相关的
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;             // 使用阶段，顶点着色器使用此资源
        
        
        
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;       // 使用阶段为 fragment shader，对于高度地形图，也可能设置在 vertex shader
        
        std::array<VkDescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();
        
        if(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw  std::runtime_error("failed to create descriptor set layout");
        }
        
        
    }
    
    // 创建渲染管线
    void createGraphicPipeline(){
        // 读取编译好的 shader 代码
        auto vertShaderCode = readFile("/Users/zhupeili/Desktop/my_program/vulkan/firstTest/vulkanTesting/vulkanTesting/shader/basic/vert.spv");
        auto fragShaderCode = readFile("/Users/zhupeili/Desktop/my_program/vulkan/firstTest/vulkanTesting/vulkanTesting/shader/basic/frag.spv");
        
        // 创建 shader module，这里不区分顶点着色器和片元着色器
        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);
        
        
        
        // assign 到 pipeline
        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT; // 指定是 vertex shader 还是 fragment shader
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main"; // 入口函数
        
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        
        VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
        
        
        
        // 添加顶点属性信息
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDescription = Vertex::getBindingDescription();          // 指定顶点缓冲区 binding 的 id，还有 stride
        auto attributeDescription = Vertex::getAttributeDescriptions();     // 指定顶点属性，相当于 pos、color 这些
        vertexInputInfo.vertexBindingDescriptionCount = 1;                  // 顶点缓冲区的个数
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescription.data();
        
        
        // input assenbly，顶点数据被装配为什么样的几何单元
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // 三角形列表，每3个顶点独立组成三角形
        inputAssembly.primitiveRestartEnable = VK_FALSE;                // 带状拓扑时候才使用，告诉这一些特殊值里面重新组装一个新的图元
        
        // 配置视口状态，
        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;    // 表示当前管线使用 1 个视口，在 vulkan 中可以同时使用多个视口，如 vr 或 分屏渲染
        viewportState.scissorCount = 1;     // 裁剪矩形的数量，世纪允许绘制的屏幕区域，以像素为单位

        
        // rasterizer
        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;  // 填充；也可以设置为线条、点
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
        
        
        // multisampling
        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional
        
        
        // color blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
        
        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional
        
        // dynamic state，为了可以在不重建管线的情况下，在命令缓冲中即时修改这些参数
        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();
        
        // 深度测试
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional


        // pipeline layout，决定着色器如何访问外部资源，如 uniform、sampler、push constant 等
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1; // descriptor set layouts的数量
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // 指向 descriptor set layouts 数组，descriptorSetLayout已经在前面 create 好了

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create pipeline layout!");
        }
        
        
        
        // 渲染管线的总配置，包含了 GPU 绘制一个图像所需要的所有状态和资源信息
        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;                                // 着色器阶段数量
        pipelineInfo.pStages = shaderStages;                        // 这里是前面配置好的 vertex shader 和 frag shader
        pipelineInfo.pVertexInputState = &vertexInputInfo;          // 顶点输入格式
        pipelineInfo.pInputAssemblyState = &inputAssembly;          // 图元装配方式
        pipelineInfo.pViewportState = &viewportState;               // 视口与裁剪矩形
        pipelineInfo.pRasterizationState = &rasterizer;             // 光栅化配置
        pipelineInfo.pMultisampleState = &multisampling;            // 多重采样
        pipelineInfo.pDepthStencilState = nullptr;                  // 深度和模版测试
        pipelineInfo.pColorBlendState = &colorBlending;             // 颜色混合
        pipelineInfo.pDynamicState = &dynamicState;                 // 动态状态
        pipelineInfo.pDepthStencilState = &depthStencil;            // 深度测试
        pipelineInfo.layout = pipelineLayout;                       // 着色器资源布局
        pipelineInfo.renderPass = renderPass;                       // 所属于的渲染通道
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;           // 是不是从现有的render pass 继承而来
        pipelineInfo.basePipelineIndex = -1;
        
        if(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
        
        // 记得释放
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        
    }
    
    // VkImage是原始图像数据（原始显存） -> VkImageView 定义了怎么访问原始图像数据 -> VkFramebuffer 是渲染目标组合，包括一个或多个VkImageView
    void createFramebuffers(){
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for(size_t i = 0; i < swapChainImageViews.size(); ++i)
        {
            std::array<VkImageView,2> attachments = {
                swapChainImageViews[i],
                depthImageView,
            };
            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = attachments.size();
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }
    
    // 创建命令池
    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }
    
    // 创建createImage的函数
    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory){
        
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = static_cast<uint32_t>(width);
        imageInfo.extent.height = static_cast<uint32_t>(height);
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.flags = 0;
        if(vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS){
            throw std::runtime_error("failed to create image");
        }
        
        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(device, image, &memRequirements);
        
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        
        if(vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS){
            throw std::runtime_error("failed to allocate image memory");
        }
//        std::cout << "binding image " << image << " to " << imageMemory << std::endl;
        vkBindImageMemory(device, image, imageMemory, 0);
    }
    
    // 转换 image 布局
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        if(newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL){
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if(hasStencilComponent(format)){
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }else{
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        }
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0; // TODO
        barrier.dstAccessMask = 0; // TODO
        
        
        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;
        if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            
            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }else {
            throw std::invalid_argument("unsupported layout transition!");
        }
        
        vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage,
                             0,
                             0, nullptr,
                             0, nullptr,
                             1, &barrier);
        
        endSingleTimeCommands(commandBuffer);
    }
    
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = {0, 0, 0};
        region.imageExtent = {
            width,
            height,
            1
        };
        
        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );
        endSingleTimeCommands(commandBuffer);
    }
    void createDepthResources(){
        VkFormat depthFormat = findDepthFormat();
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
        transitionImageLayout(depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        
    }
    
    // 从一组里面找到锁需要的 format
    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features){
        for(VkFormat format:candidates){
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice,format, &props);
            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }

        }
        throw std::runtime_error("failed to find supported format!");
    }
    
    // 选择一个深度 format
    VkFormat findDepthFormat(){
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT}, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }
    
    bool hasStencilComponent(VkFormat format){
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }
    
    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load("/Users/zhupeili/Desktop/my_program/vulkan/firstTest/vulkanTesting/vulkanTesting/textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;
        
        
        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        std::cout << "image: " << texWidth << " " << texHeight << " " << texChannels << std::endl;
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data); // 映射一块 gpu 内存到指针 void中
        memcpy(data, pixels, imageSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        stbi_image_free(pixels);
        
        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);
        
        // 转换布局，从 undefined 到 transfer dst
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        // 拷贝缓存数据
        copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        // 转换布局，从 transfer dst 到 shader read
        transitionImageLayout(textureImage, VK_FORMAT_R8G8B8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        
        
        // 释放 staging
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    
    void createTextureImageView(){
        textureImageView = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
    }
    
    void createTextureSampler(){
        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        
        if (vkCreateSampler(device, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }
    }
    
    void createVertexBuffer(){
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        
        // 把数据从 cpu 拷贝到 GPU
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, vertices.data(), (size_t) bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
        
        // 释放 staging
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }
    

    void createIndexBuffer(){
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
        
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
        
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }


    
    void createUniformBuffers(){
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i){
            createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
        
    }
    
    void createDescriptorPool() {
        std::array<VkDescriptorPoolSize,2> poolSizes{};
//        VkDescriptorPoolSize poolSize{};
        // uniform 的
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
        
    }
   
    void createDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();
        
        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
        
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i){
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);
            
            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;
            
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            
            // 先是 uniform
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;
            
            // 然后是纹理采样 sampler
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
    }

    // 把 “缓冲区对象 + 需要到 GPU 内存“ 绑定在一起，让 CPU/GPU 都能访问，从而可以用来存储顶点、索引、uniform 等数据
    // usage —— 缓冲区用途，如 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT 顶点缓冲
    //                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT 索引缓冲
    //                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT uniform 缓冲
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory){
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // 仅被一个队列族使用
        
        if(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS){
            throw std::runtime_error("failed to create buffer!");
        }
        
        
        // 由驱动告知当前 vertex buffer 需要的 GPU 内存类型
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer, &memRequirements);
        
        
        // 查询并分配内存
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // 找到一种同时满足可被 CPU 映射访问，且保证 CPU/GPU写入一致性的内存
        
        if(vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate vertex buffer memory");
            
        }
        
        // 把内存和 buffer 绑定
        vkBindBufferMemory(device, buffer, bufferMemory, 0);
        
    }
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    // buffer 拷贝函数，把一个缓冲区的数据拷贝到另一个缓冲区（如从 cpu 可以访问的 staging buffer 拷贝到 GPU 只读到 vertex buffer）
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();
        
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0; // Optional
        copyRegion.dstOffset = 0; // Optional
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        
        endSingleTimeCommands(commandBuffer);
    }
    
    
    
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        
        for(uint32_t i=0; i<memProperties.memoryTypeCount; ++i){
            if((typeFilter & (1 << i)) && ((memProperties.memoryTypes[i].propertyFlags & properties) == properties))
            {
                return i;
            }
        }
        
        throw std::runtime_error("failed to find suitable memory type!");
        return 0;
    }
    
    void createCommandBuffer(){
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
    
    
    // 把 command 写进到 command buffer
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;
        if(vkBeginCommandBuffer(commandBuffer, &beginInfo)!=VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
        
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapChainExtent;
        
        // 清除attachments里面的信息
        std::array<VkClearValue,2> clearValues{};
        clearValues[0].color = {{1.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();
        
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        
        
        // 在创建pipeline到时候，我们设置viewports and scissors为 dynamica了，所以，这里需要手动设置一下他们的值
        VkViewport viewport{};  // 视口的大小
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) swapChainExtent.width;
        viewport.height = (float) swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        
        VkRect2D scissor{};     // 对视图进行裁剪
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        
        // 手动绑定 vertex buffer
        VkBuffer vertexBuffers[] = {vertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
        
        // 绑定 index buffer
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
        
        // 绑定 uniform 数据
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        
        // 直接绘制顶点
//        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);// vertex count; instance count; first vertex; first instance
        // 使用 index buffer 来绘制
        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        
        // 结束 render pass
        vkCmdEndRenderPass(commandBuffer);
        
        if(vkEndCommandBuffer(commandBuffer)!= VK_SUCCESS)
        {
            throw std::runtime_error("failed to end render pass");
        }
    }
    
    void createSyncObjects(){
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        
        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;// 初始化设置为 signaled ，避免循环一开始等不到 fence，陷入死循环
        
        for(int i=0; i<MAX_FRAMES_IN_FLIGHT; ++i)
        {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }
        }
    }

    // 每帧都更新
    void updateUniformBuffer(uint32_t currentImage){
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
        
        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        
        ubo.proj[1][1] *= -1;
        
        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    // 渲染一帧的流程如下：
    // 等前一帧完成
    // 从 swap chain 中获取 image
    // 往 command buffer 写入绘制命令
    // 显示 swap chain 的 image
    void drawFrame(){
        // 等待 fence完成
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        
        
        // 从 swap chain 中获取一个 image
        uint32_t imageIndex;
        
        // 从交换链中取出一张图片来进行渲染，如果返回的结果表示 swapchain 不适合，就重新建立 swap chain
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if(VK_ERROR_OUT_OF_DATE_KHR == result)
        {
            
            recreateSwapChain();
            return;
        }else if(VK_SUCCESS != result && VK_SUBOPTIMAL_KHR != result)
        {
            throw std::runtime_error("fail to acquire swap chain image");
        }
        // 更新 uniform 数据
        updateUniformBuffer(currentFrame);
        
        // fence触发后，记得 reset
        // 放在重建 swap chain 之后，
        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        
        
        
        
        // 使用 imageIndex 进行图像绘制
        vkResetCommandBuffer(commandBuffers[currentFrame],0);
        recordCommandBuffer(commandBuffers[currentFrame], imageIndex);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        
        
        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;
        
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        
        // 把渲染好的图像显示到屏幕上
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = {swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        
        
        // 把图片显示到屏幕
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if(VK_ERROR_OUT_OF_DATE_KHR == result  || VK_SUBOPTIMAL_KHR == result || framebufferResized)
        {
            framebufferResized = false; // reset this
            recreateSwapChain();
        }else if(VK_SUCCESS != result){
            throw std::runtime_error("fail to present image");
        }
        
        currentFrame = (currentFrame + 1 ) % MAX_FRAMES_IN_FLIGHT;
        
    }
    
    
    
    // 根据 spir-v 代码，创建着色器 module，这里不区分 vertex shader 和 fragment shader，后续阶段才区分
    VkShaderModule createShaderModule(const std::vector<char>& code){
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }
    
    
    
    // 具备 swap chain 能力后，还需要确定具体的配置
    // 主要包括下面这些：
    // surface format (color depth)
    // Presentation mode(conditions for "swapping" images to the screen)
    // swap extent (resolution of images in swap chain)
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availiableFormats){
        for(const auto& format:availiableFormats)
        {
            if(format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return format;
            }
        }
        return availiableFormats[0];
    }
    
    
    // 选择一个 present mode
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }
        //  默认情况下会支持的 present mode
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    
    //swap extent，相当于是分辨率
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities){
        if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
            return capabilities.currentExtent;
        }
        else{
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            
            VkExtent2D actualExtent={
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
            // minImageExtent 和 maxImageExtent 是允许的最小和最大
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            
            return actualExtent;
        }
    }
    
    // 查看一些 swap chain 的信息
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device)
    {
        SwapChainSupportDetails details;
        // min/max image num 和 image 宽高
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);
        
        // format
        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        
        // mode
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        
        return details;
    }
    


    // 判断一个 device 是否支持特定的 queue family，来判断显卡是不是合适
    bool isDeviceSuitable(VkPhysicalDevice& device)
    {
        
        QueueFamilyIndices indices = findQueueFamilies(device);
        
        
        // 是不是支持一些如 swap chain 的 extension
        bool extensionSupport = checkDeviceExtensionSupport(device);
        
        // 看看 swap chain 的能力够不够
        bool swapChainAdequate = false;
        if(extensionSupport)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        
        // 查看支持的 deviceFeatures
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        
        return indices.isComplete() && extensionSupport && swapChainAdequate && supportedFeatures.samplerAnisotropy;
        
    }
    
    
    // 看看显示卡是不是支持特定的功能，如 swap chain（用来绘制图像的）
    bool checkDeviceExtensionSupport(VkPhysicalDevice device)
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
        
        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());
        
        std::cout << "current physical device " << device << " support extensions num:" << extensionCount << std::endl;
        
        // 这里是我们想要配置的 extension，deviceExtension 是我们的前面设置的全局变量
        std::set<std::string> requiredExtensions(deviceExtensions.begin(),deviceExtensions.end());
        // 判断一下我们的 requiredExtensions 里面，是不是有放入不支持的 extension
        for(const auto& extension:availableExtensions)
        {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
        
    }
    
    
    // queue 能力核查
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice& device)
    {
        QueueFamilyIndices indices;
        
        uint32_t queueFamilyCount;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
        
        std::cout << "current physical device supported queue family num: " << queueFamilyCount << std::endl;
        
        int i = 0;
        for(const auto& queueFamily:queueFamilies)
        {
            // 检查 present 能力
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            
            if(presentSupport)
            {
                indices.presentFamily = i;
            }
            
            // queue 能力核查
            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) // 支持图形绘制命令（渲染管线）
            {
                indices.graphicsFamily = i;
            }
            if(indices.isComplete())
            {
                break;
            }
            i++;
        }
        
        
        return indices;
    }
    
    
    
    
    // 获取需要的 extensions,如果配置了 validation，也需要添加对应的 extension
    std::vector<const char*> getRequiredExtensions(bool showSupport=false)
    {
        if(showSupport)
        {
            // 打印一下当前支持的 extension
            uint32_t extensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> extensions(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

            std::cout << "supported extension: " << extensionCount << std::endl;
            uint32_t cnt = 0;
            for(auto& extension:extensions)
            {
                std::cout <<cnt++ << "\t"<< extension.extensionName << std::endl;
            }
        }
        
        // glfw窗口所需要的实例拓展
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        std::vector<const char*> requiredExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount); // 构造函数，数组的起始指针和数组的结束指针
        
#if __APPLE__
        requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME); // macos 需要这个
#endif
        
        // debug用
        if(enableValidationLayers)
        {
            requiredExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        
        return  requiredExtensions;
    }
    
    // 如果开启了 enablevalidation，检查一下配置的layer 是不是都支持
    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);   // 读取支持的layer数量信息
        
        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
        
        std::cout << "support validation layers: " << layerCount << std::endl;
        uint32_t cnt=0;
        for(auto& layer:availableLayers)
        {
            std::cout <<cnt++ <<"\t"<< layer.layerName << std::endl;
        }
        
        // 检查一下要使用的 validation layer 在不在支持的列表里面
        for(const char* layer:validationLayers)
        {
            bool layerFound = false;
            for(const auto& availLayer:availableLayers)
            {
                if(strcmp(layer, availLayer.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }
            
            if(!layerFound)
            {
                return false;
            }
        }
        return true;
    }
    
    
    // 文件读取函数
    static std::vector<char> readFile(const char* fileName)
    {
        std::ifstream inF(fileName, std::ios::binary | std::ios::ate);
        if(!inF.is_open())
        {
            std::string msg = std::string("open ") + fileName + std::string(" fail\n");
            throw std::runtime_error(msg.c_str());
        }
        
        size_t fileSize = (size_t)inF.tellg();
        std::vector<char> buffer(fileSize);
        inF.seekg(0);
        inF.read(buffer.data(), fileSize);
        
        inF.close();
        return buffer;
    }
    
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
    
    
};

int main()
{
    std::filesystem::path cwd = std::filesystem::current_path();
    
    std::cout << "hello: " <<cwd<< std::endl;
    HelloTriangleApplication app;
    try{
        app.run();
    }catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

#endif
