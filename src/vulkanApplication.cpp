#include "vulkanApplication.h"


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

//-------------------------------- VulkanApplication 类 ----------------------------

//-------------------------------- 对外函数
VulkanApplication::VulkanApplication(const std::string& appName,int width, int height):m_applicationName(appName),m_windowWidth(width), m_windowHeight(height) {
    
}

void VulkanApplication::run()
{
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

bool VulkanApplication::checkValidationLayerSupport(bool show){
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);   // 读取支持的layer数量信息
    
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    
    if(show){
        std::cout << "support validation layers: " << layerCount << std::endl;
        uint32_t cnt=0;
        for(auto& layer:availableLayers)
        {
            std::cout <<cnt++ <<"\t"<< layer.layerName << std::endl;
        }
    }
    
    
    // 检查一下要使用的 validation layer 在不在支持的列表里面
    for(const char* layer:m_validationLayers)
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

// 获取需要的 extensions,如果配置了 validation，也需要添加对应的 extension
std::vector<const char*> VulkanApplication::getRequiredExtensions(bool showSupport)
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
    if(m_enableValidationLayers)
    {
        requiredExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    
    return  requiredExtensions;
}

// ------------------------------- 一些判断函数
QueueFamilyIndices VulkanApplication::findQueueFamilies(VkPhysicalDevice& device)
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

bool VulkanApplication::checkDeviceExtensionSupport(VkPhysicalDevice device){
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

SwapChainSupportDetails VulkanApplication::querySwapChainSupport(VkPhysicalDevice device)
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

bool VulkanApplication::isDeviceSuitable(VkPhysicalDevice& device)
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

VkSampleCountFlagBits VulkanApplication::getMaxUsableSampleCount(){
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
    VkSampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & VK_SAMPLE_COUNT_64_BIT) { return VK_SAMPLE_COUNT_64_BIT; }
    if (counts & VK_SAMPLE_COUNT_32_BIT) { return VK_SAMPLE_COUNT_32_BIT; }
    if (counts & VK_SAMPLE_COUNT_16_BIT) { return VK_SAMPLE_COUNT_16_BIT; }
    if (counts & VK_SAMPLE_COUNT_8_BIT) { return VK_SAMPLE_COUNT_8_BIT; }
    if (counts & VK_SAMPLE_COUNT_4_BIT) { return VK_SAMPLE_COUNT_4_BIT; }
    if (counts & VK_SAMPLE_COUNT_2_BIT) { return VK_SAMPLE_COUNT_2_BIT; }

    return VK_SAMPLE_COUNT_1_BIT;
}

VkSurfaceFormatKHR VulkanApplication::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availiableFormats){
    for(const auto& format:availiableFormats)
    {
        if(format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return format;
        }
    }
    return availiableFormats[0];
}

VkPresentModeKHR VulkanApplication::chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
    }
    //  默认情况下会支持的 present mode
    return VK_PRESENT_MODE_FIFO_KHR;
}

//swap extent，相当于是分辨率
VkExtent2D VulkanApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities){
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
    
void VulkanApplication::recreateSwapChain()
{
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
    vkDeviceWaitIdle(m_device);
    
    cleanupSwapChain();
    
    createSwapChain();
    createImageViews();
    createColorResources();
    createDepthResources();
    createFramebuffers();
    
}

VkFormat VulkanApplication::findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features){
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

std::vector<char> VulkanApplication::readFile(const char *filePath)
{
    std::ifstream inF(filePath, std::ios::binary | std::ios::ate);
    if(!inF.is_open())
    {
        std::string msg = std::string("open ") + filePath + std::string(" fail\n");
        throw std::runtime_error(msg.c_str());
    }
    
    size_t fileSize = (size_t)inF.tellg();
    std::vector<char> buffer(fileSize);
    inF.seekg(0);
    inF.read(buffer.data(), fileSize);
    
    inF.close();
    return buffer;
}

VkShaderModule VulkanApplication::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module!");
    }
    return shaderModule;
}
uint32_t VulkanApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
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
//-------------------------------- 内部函数

void VulkanApplication::setupDebugMessenger()
{
    if(!m_enableValidationLayers)return;
    
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
    
    if(CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to setup debug messenger!");
    }
    
}

void VulkanApplication::initWindow(){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    
    
    // 创建窗口
    window = glfwCreateWindow(m_windowWidth, m_windowHeight, m_applicationName.c_str(), nullptr, nullptr);
    
    // glfw 可以设置一个用户自定义的指针，所以，我们可以把对象的this指针保存下来
    glfwSetWindowUserPointer(window, this);
    
    // 窗口大小发生变化时候到回调函数
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VulkanApplication::framebufferResizeCallback(<#GLFWwindow *window#>, <#int width#>, <#int height#>){
    auto app = reinterpret_cast<VulkanApplication*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanApplication::initVulkan(){
    createInstance();       // 启用 extensions，主要包 glfw 需要的、debug、还有不同操作系统相关的
    setupDebugMessenger();  // 启用 debug 功能，配合 validation，可以打印一些信息
    createSurface();        // 创建 vulkan 表面，vulkan与窗口系统连接的桥梁，负责呈现到屏幕
    pickPhysicalDevice();   // 选择一个图形卡，不同的 physical device，支持不同的extensions，这里选择支持 swap chain的物理设备
    createLogicalDevice();  // 创建 logical device，获取 graphic queue 和 present queue
    createSwapChain();      // 创建交换链，
    createImageViews();     // 为每个交换链中的数据创建一个 image views
    createRenderPass();     // 附件、subpass，render pass
    createDescriptorSetLayout();    // 创建描述符集布局，定义了着色器如何访问资源（uniform 缓冲区、纹理等）的接口“规范”
    createGraphicPipeline("vulkanTesting/shader/basic/vert.spv","vulkanTesting/shader/basic/frag.spv");        // 创建渲染管线
    createCommandPool();            // 创建命令池
    createColorResources();         // 为 msaa 创建对应的资源，主要包括 image、imageview 还有对应的 memory
    createDepthResources();         // 深度测试相关的内容
    createFramebuffers();           // 创建帧缓存，定义一个帧里面有多少个 view port
    createTextureImage();
    createTextureImageView();
    createTextureSampler();
    loadModel();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffer();
    createSyncObjects();
}

void VulkanApplication::mainLoop()
{
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(m_device);
}

void VulkanApplication::cleanupSwapChain(){
    // 释放 framebuffer
    for(auto framebuffer:swapChainFramebuffers){
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }
    // 释放 image view
    for (auto imageView : swapChainImageViews) {
       vkDestroyImageView(m_device, imageView, nullptr);
    }
    // 释放 swap chain
    vkDestroySwapchainKHR(m_device, swapChain, nullptr);
}

void VulkanApplication::cleanup(){
    cleanupSwapChain();

    // 释放纹理相关的内存和变量
    vkDestroyImage(m_device, textureImage, nullptr);
    vkDestroyImageView(m_device, textureImageView, nullptr);
    vkFreeMemory(m_device, textureImageMemory, nullptr);
    
    // 释放深度测试相关的资源
    vkDestroyImage(m_device, depthImage, nullptr);
    vkDestroyImageView(m_device, depthImageView, nullptr);
    vkFreeMemory(m_device, depthImageMemory, nullptr);
    
    // 释放超采样相关的资源
    vkDestroyImage(m_device, colorImage, nullptr);
    vkDestroyImageView(m_device, colorImageView, nullptr);
    vkFreeMemory(m_device, colorImageMemory, nullptr);
    
    // 释放 VkSampler
    vkDestroySampler(m_device, textureSampler, nullptr);
    
    // 释放 uniform map
    for(size_t i=0; i<MAX_FRAMES_IN_FLIGHT; ++i){
        vkDestroyBuffer(m_device, uniformBuffers[i], nullptr);
        vkFreeMemory(m_device, uniformBuffersMemory[i], nullptr);
    }
    
    vkDestroyDescriptorPool(m_device, descriptorPool, nullptr);
    //清除 layout descriptorr
    vkDestroyDescriptorSetLayout(m_device, descriptorSetLayout, nullptr);
    
    // 释放 vertex buffer
    vkDestroyBuffer(m_device, vertexBuffer, nullptr);
    vkFreeMemory(m_device, vertexBufferMemory, nullptr);
    
    // 释放 index buffer
    vkDestroyBuffer(m_device, indexBuffer, nullptr);
    vkFreeMemory(m_device, indexBufferMemory, nullptr);
    
    
    // 释放用于同步的 semaphore 和 fence
    for(int i=0; i<MAX_FRAMES_IN_FLIGHT; ++i){
        vkDestroySemaphore(m_device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(m_device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(m_device, inFlightFences[i], nullptr);
    }

    
    // 释放 command pool
    vkDestroyCommandPool(m_device, commandPool, nullptr);
    
    
    // 释放 graphics pipeline
    vkDestroyPipeline(m_device, graphicsPipeline, nullptr);
    
    // 释放 pipeline layout
    vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);
    // 释放 renderPass
    vkDestroyRenderPass(m_device, renderPass, nullptr);
    
    
    // 释放 logical device
    vkDestroyDevice(m_device, nullptr);
    
    if(m_enableValidationLayers)
    {
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
    }
    
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
}

void VulkanApplication::createInstance(bool show){
    // app名字、版本信息
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = m_applicationName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "NO ENGINE";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;
    
    // instance
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    
    // layer 配置，看看是不是配置的layer，当前硬件存在不支持的情况
    if(m_enableValidationLayers && !checkValidationLayerSupport(show))
    {
        throw std::runtime_error("validation layer requisted, but not supported\n");
    }
    
    if(m_enableValidationLayers)
    {
        createInfo.enabledLayerCount = (uint32_t)m_validationLayers.size();
        createInfo.ppEnabledLayerNames = m_validationLayers.data();
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


void VulkanApplication::createSurface(){

    if( glfwCreateWindowSurface(instance, window, nullptr, &surface)!= VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface with the error");
    }
}

void VulkanApplication::pickPhysicalDevice(bool show){
    
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
            msaaSamples = getMaxUsableSampleCount();
            std::cout << "max sample point per pixel: " << msaaSamples << std::endl;
            break;
        }
    }
    
    if(VK_NULL_HANDLE == physicalDevice)
    {
        throw std::runtime_error("failed to find suitable GPU");
    }
    
}

void VulkanApplication::createLogicalDevice(){
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
    if(m_enableValidationLayers)
    {
        createInfo.enabledLayerCount = (uint32_t)(m_validationLayers.size());
        createInfo.ppEnabledLayerNames = m_validationLayers.data();
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
    if(vkCreateDevice(physicalDevice, &createInfo, nullptr, &m_device) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create logical device!\n");
    }
    
    // 读取 graphics queue
    vkGetDeviceQueue(m_device, indices.graphicsFamily.value(), 0, &graphicsQueue);
    // 读取 present queue
    vkGetDeviceQueue(m_device, indices.presentFamily.value(), 0, &presentQueue);
}

void VulkanApplication::createSwapChain(){
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);       // 查询物理设备的显示信息
    
    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);   // 选择颜色格式
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);    // 呈现模式
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);                    // 分辨率，通常是窗口大小
    
    // 确定 swap chain 里面有多少数量的 images
    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    std::cout << "image count in swap chain: " << imageCount << std::endl;
    
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
    
    
    if (vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }
    
    // swapChain 创建完成后，可以去获取一些 image 中的内容了，同样，我们需要先获取数量
    vkGetSwapchainImagesKHR(m_device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, swapChain, &imageCount, swapChainImages.data());
    
    std::cout << "num of image in swap chain : " << imageCount << std::endl;
    
    // 保存一下当前的图像格式、分辨率
    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
    std::cout <<"width and height: "<< swapChainExtent.width <<" "<< swapChainExtent.height << std::endl;
}


VkImageView VulkanApplication::createImageView(VkImage image, VkFormat format, uint32_t mipLevels, VkImageAspectFlags aspectFlags) {
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.subresourceRange.levelCount = mipLevels;

    VkImageView imageView;
    if (vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image view!");
    }

    return imageView;
}

void VulkanApplication::createImageViews(){
    swapChainImageViews.resize(swapChainImages.size());
    // 遍历 ImageViews
    for(size_t i = 0; i < swapChainImages.size(); ++i){
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, 1, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void VulkanApplication::createRenderPass()
{
    // 附件管理，主要包括颜色附件、引用附件
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    
    
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;       // what to do before rendering
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;     // what to do after rendering,这里配置的是渲染后的contents会存在内存里面，可以后面再读取
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    
    // 下面是 msaa 相关的内容
    colorAttachment.samples = msaaSamples;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL; // 渲染后的布局，这里设置为颜色附件布局，其他值如 VK_IMAGE_LAYOUT_PRESENT_SRC_KHR 表示呈现到屏幕布局
    
    
    // 下面是子通道定义，颜色引用附件
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    // 深度附件
    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = msaaSamples;
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
    
    // 启用 msaa 的情况下，需要将渲染后的图像 resolve 之后才能呈现到屏幕上面
    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = swapChainImageFormat;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    // msaa 对应的color attachment 的附件引用
    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    
    // subpass
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
    
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    // 创建渲染通道
    std::array<VkAttachmentDescription,3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};
    
    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = attachments.size();                 // 颜色附件，深度附件或其他模版附件的数量
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    

    if (vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void VulkanApplication::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;   // 允许哪些着色器阶段访问这个 binding
    
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
    
    if(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS)
    {
        throw  std::runtime_error("failed to create descriptor set layout");
    }
    
}

void VulkanApplication::createGraphicPipeline(const char *vertSpv, const char *fragSpv)
{
    // 读取编译好的 shaer 代码
    auto vertShaderCode = readFile(vertSpv);
    auto fragShaderCode = readFile(fragSpv);
    
    
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
    rasterizer.cullMode = VK_CULL_MODE_NONE;        // 关闭面剔除
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f; // Optional
    rasterizer.depthBiasClamp = 0.0f; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
    
    
    // multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;           // 配置为当前显卡支持的最大的 mass 采样点
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

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
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
    
    if(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create graphics pipeline!");
    }
    
    // 记得释放
    vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
    
}


void VulkanApplication::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for(size_t i = 0; i < swapChainImageViews.size(); ++i)
    {
        std::array<VkImageView,3> attachments = {
            colorImageView,
            depthImageView,
            swapChainImageViews[i],
        };
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanApplication::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
    
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

    if (vkCreateCommandPool(m_device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void VulkanApplication::createImage(uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image, VkDeviceMemory &imageMemory)
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = static_cast<uint32_t>(width);
    imageInfo.extent.height = static_cast<uint32_t>(height);
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = numSamples;         // 如果开启了 msaa，这里就是配置每个像素采样点的数量
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;
    
    if(vkCreateImage(m_device, &imageInfo, nullptr, &image) != VK_SUCCESS){
        throw std::runtime_error("failed to create image");
    }
    
    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, image, &memRequirements);
    
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
    
    if(vkAllocateMemory(m_device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS){
        throw std::runtime_error("failed to allocate image memory");
    }
//        std::cout << "binding image " << image << " to " << imageMemory << std::endl;
    vkBindImageMemory(m_device, image, imageMemory, 0);
}

VkCommandBuffer VulkanApplication::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanApplication::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(m_device, commandPool, 1, &commandBuffer);
}

void VulkanApplication::transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels)
{
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
    barrier.subresourceRange.levelCount = mipLevels;
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

void VulkanApplication::copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
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
