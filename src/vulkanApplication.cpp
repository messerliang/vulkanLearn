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
    createGraphicPipeline();        // 创建渲染管线
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
