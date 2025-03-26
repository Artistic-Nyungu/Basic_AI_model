#include <iostream>
#include <cmath>
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// Hyperparameters
const int NODES_PER_LAYER[] = {784, 6, 4, 6, 10};
const float LEARNING_RATE = 0.01;
const int MAX_ITERATIONS = 500;

/// Utility function to get the magnitude of an array
/// arr - array to be normalized
/// length - the length of the array
/// returns - a float representing the magnitude
float magnitude(const float* arr, int length){
    float squareSum = 0.0f;
    for (int i=0; i<length; i++) {
        squareSum += arr[i] * arr[i];
    }
    return std::sqrt(squareSum);
}

/// Utility function to normalize an array
/// arr - array to be normalized
/// length - the length of the array
/// returns - an array with the same length as the original (Memory has to be managed)
float* normalize(const float* arr, int length){
    float mag = magnitude(arr, length);
    float* normalized = new float[length];
    for (int i=0; i<length; i++) {
        normalized[i] = arr[i] / mag;
    }

    return normalized;
}

struct range{
    int begin;
    int end;
};

struct forwardingLayer{
    range nodes;
    range weights;
    range biases;
};

int main() {
    // Initialize library
    if(!glfwInit())
    {
        std::cerr << "GLFW could not start\n";
        exit(-1);  
    }

    // Instantiate window
    auto _window = glfwCreateWindow(640, 480, "OpenGL Programming Application", NULL, NULL);
    if(!_window)
    {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        exit(-1);
    }
    
    // Make window's context current, has to be before loading glad
    glfwMakeContextCurrent(_window);
    
    // Initialize ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(_window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();

    // Set callback function for when the window size is changed
    glfwSetFramebufferSizeCallback(_window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });


    // NN
    int _nNodes = 0, _nWeights = 0, _nBiases = 0;
    int nplLength = sizeof(NODES_PER_LAYER)/sizeof(int);
    forwardingLayer* _forwardingLayers = new forwardingLayer[nplLength - 1];
    for(int i=0; i < nplLength; i++){
        _nNodes += NODES_PER_LAYER[i];  // Count all the nodes
        if(i < nplLength - 1){
            _nWeights += NODES_PER_LAYER[i] * NODES_PER_LAYER[i+1]; // Count all the weights
            _nBiases += NODES_PER_LAYER[i+1]; // Count all the biases
            // Add ranges for forwarding layers
            // Layer nodes end() = sum of npy(i) from i=0 to i=current layer, begin() = end() - npy(current layer)
            // Weights end() = sum of ( npy(i) * npy(i+1) ) from i=0 to i=current layer, i < nLayers , begin() = end() - npy(current layer) * npy(current layer + 1)
            // Biases end() = sum of npy(i+1) from i=0 to i=current layer, i < nLayers, begin() = end() - npy(current layer + 1)
            _forwardingLayers[i] = {
                {_nNodes, _nNodes - NODES_PER_LAYER[i]},
                {_nWeights, _nWeights - NODES_PER_LAYER[i] * NODES_PER_LAYER[i+1]},
                {_nBiases, _nBiases - NODES_PER_LAYER[i+1]}
            };
        }
    }


    // All layer features will be mapped to a 1D array
    // Features of one layer will be sequential until the nth of the layer,
    // then the next will belong to the following layer
    float* _nodes = new float[_nNodes]; 
    float* _weights = new float[_nWeights];
    float* _biases = new float[_nBiases]{0};

    // Initialize weights to random values between -5 & 5
    for(int i=0; i<_nWeights; i++)
        _weights[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;

    ImVec4 clearColor = {};
    while(!glfwWindowShouldClose(_window))
    {
        glClearColor(clearColor.x, clearColor.y, clearColor.z, clearColor.w);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui::NewFrame();

        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImVec2 wSize = ImGui::GetContentRegionAvail();

        glViewport(pos.x, pos.y, wSize.x, wSize.y);

        // Rendering commands here
        ImGui::Begin("Basic AI Model");
            ImGui::Text("Hello World");
        ImGui::End();


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        /* Swap front and back buffers */
        glfwSwapBuffers(_window);

        glfwPollEvents();
    }

    // Clean
    delete[] _nodes;
    delete[] _weights;
    delete[] _biases;
    delete[] _forwardingLayers;
    _nodes = _weights = _biases = nullptr;
    _forwardingLayers = nullptr;

    // Clean glfw
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    return 0;
}
