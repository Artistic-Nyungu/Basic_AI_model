#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

// Hyperparameters
// const int NODES_PER_LAYER[] = {784, 6, 4, 6, 10};
const int NODES_PER_LAYER[] = {4, 3, 4, 2, 3};
const float LEARNING_RATE = 0.01;
const int MAX_ITERATIONS = 500;

const char* DATA_FILENAME = "./data/iris/iris.data"; // Path to the dataset

struct shader
{
    const char* sourcePath;
    unsigned int type;
};

const shader SHADERS[] = {{"./shaders/feedforward.comp", GL_COMPUTE_SHADER},
                        // {"./shaders/backprop.comp", GL_COMPUTE_SHADER},
                        {"./shaders/frag.glsl", GL_FRAGMENT_SHADER},
                        {"./shaders/vert.glsl", GL_VERTEX_SHADER}};

// Vertices for the quad (2 triangles) that we are drawing the visualization on
const float VERTS[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
    -1.0f,  1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f
};

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
    range neurons;
    range weights;
    range biases;
    int srcNeurons;   // Neurons in the previous (source) layer
    int dstNeurons;   // Neurons in the current (destination) layer
};


int main() {
    srand(time(nullptr));

    // Initialize library
    if(!glfwInit())
    {
        std::cerr << "GLFW could not start\n";
        exit(-1);  
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

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

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwTerminate();
        exit(-1);
    }
    
    // Initialize ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(_window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
    ImGui_ImplOpenGL3_Init();

    #ifdef _DEBUG
        // Check environment
        printf("Renderer: %s\n", glGetString(GL_RENDER));
        printf("Vender: %s\n", glGetString(GL_VENDOR));
        printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    #endif


    // Set callback function for when the window size is changed
    glfwSetFramebufferSizeCallback(_window, [](GLFWwindow* window, int width, int height) {
        glViewport(0, 0, width, height);
    });

    // Register shader
    unsigned int _renderModule, _computeModule;
    if(true){
        _renderModule = glCreateProgram();
        _computeModule = glCreateProgram();

        for(shader sh: SHADERS){
            std::ifstream shaderFile(sh.sourcePath);
            if (!shaderFile.is_open()) {
                std::cerr << "Failed to open shader file: " << sh.sourcePath << std::endl;
                continue;
            }
            std::string contents((std::istreambuf_iterator<char>(shaderFile)), std::istreambuf_iterator<char>());
            const char* shaderSource = contents.c_str();

            // Compile shader
            auto shader = glCreateShader(sh.type);
            glShaderSource(shader, 1, &shaderSource, nullptr);
            glCompileShader(shader);

            int success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if(!success){
                char infoLog[512];
                glGetShaderInfoLog(shader, 512, nullptr, infoLog);
                std::cerr << "ERROR::SHADER_COMPILATION_FAILED\n" << infoLog << std::endl;
                
                #ifdef _DEBUG
                    printf("Shader source (%s): \n%s\n", glGetString(sh.type), shaderSource);
                #endif
            }
            sh.type == GL_COMPUTE_SHADER ? glAttachShader(_computeModule, shader) : glAttachShader(_renderModule, shader);
            glDeleteShader(shader);
        }

        glLinkProgram(_computeModule);
        glLinkProgram(_renderModule);

        int success;
        glGetProgramiv(_computeModule, GL_LINK_STATUS, &success);
        if(!success){
            char infoLog[512];
            glGetProgramInfoLog(_computeModule, 512, nullptr, infoLog);
            std::cerr << "ERROR::COMPUTE_PROGRAM_LINKING_FAILED\n" << infoLog << std::endl;
        }
        
        glGetProgramiv(_renderModule, GL_LINK_STATUS, &success);
        if(!success){
            char infoLog[512];
            glGetProgramInfoLog(_renderModule, 512, nullptr, infoLog);
            std::cerr << "ERROR::RENDER_PROGRAM_LINKING_FAILED\n" << infoLog << std::endl;
        }
    }

    // Set renderer Uniforms
    // glUniform1f(glGetUniformLocation(_renderModule, "minValNeurons"), 0.f);
    // glUniform1f(glGetUniformLocation(_renderModule, "maxValNeurons"), 1.f);
    
    // NN
    int _nTargets = 0;
    std::unordered_map<std::string, int> _targets;
    int _nNeurons = 0, _nWeights = 0, _nBiases = 0;
    int nplLength = sizeof(NODES_PER_LAYER)/sizeof(int);
    forwardingLayer* _forwardingLayers = new forwardingLayer[nplLength - 1];
    for(int i=0; i < nplLength; i++){
        _nNeurons += NODES_PER_LAYER[i];  // Count all the neurons
        if(i < nplLength - 1){
            int srcNeurons = NODES_PER_LAYER[i];
            int dstNeurons = NODES_PER_LAYER[i+1];
            int weights = srcNeurons * dstNeurons;

            _forwardingLayers[i] = {
                {_nNeurons, _nNeurons + dstNeurons},
                {_nWeights, _nWeights + weights},
                {_nBiases, _nBiases + dstNeurons},
                srcNeurons,
                dstNeurons
            };
            _nWeights += weights; // Count all the weights
            _nBiases += dstNeurons; // Count all the biases
        }
    }


    // All layer features will be mapped to a 1D array
    // Features of one layer will be sequential until the nth of the layer,
    // then the next will belong to the following layer
    float* _neurons = new float[_nNeurons]; 
    float* _weights = new float[_nWeights];
    float* _biases = new float[_nBiases]{0};

    // Arrays for storing the weight and bias gradients
    float* _weightGradients = new float[_nWeights]{0};
    float* _biasGradients = new float[_nBiases]{0};

    // Initialize weights to random values between -5 & 5
    float minWeight = FLT_MAX, maxWeight = FLT_MIN;
    for(int i=0; i<_nWeights; i++){
        _weights[i] = (float) rand() / (float) RAND_MAX * 10.0f - 5.0f;
        if(minWeight > _weights[i])
            minWeight = _weights[i];
        if(maxWeight < _weights[i])
            maxWeight = _weights[i];
    }

    #ifdef _DEBUG
    // Initialize neurons to random values between -5 & 5 (NOT PRACTICAL), only for debugging visualization
    float minNeuron = FLT_MAX, maxNeuron = FLT_MIN;
    for(int i=0; i<_nNeurons; i++){
        _neurons[i] = (float) rand() / (float) RAND_MAX * 10.0f - 5.0f;
        if(minNeuron > _neurons[i])
            minNeuron = _neurons[i];
        if(maxNeuron < _neurons[i])
            maxNeuron = _neurons[i];
    }

    glUseProgram(_renderModule);
    glUniform1f(glGetUniformLocation(_renderModule, "minValNeurons"), minNeuron);
    glUniform1f(glGetUniformLocation(_renderModule, "maxValNeurons"), maxNeuron);
    #endif

    glUseProgram(_renderModule);
    glUniform1f(glGetUniformLocation(_renderModule, "minValWeights"), minWeight);
    glUniform1f(glGetUniformLocation(_renderModule, "maxValWeights"), maxWeight);
    glUniform1i(glGetUniformLocation(_renderModule, "layersCount"), nplLength - 1);


    // Copy neurons, weights & biases to SSBO
    const int nBuffers = 6;
    unsigned int _SSBOs[nBuffers];
    glGenBuffers(nBuffers, _SSBOs);
    // Neurons
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, _nNeurons * sizeof(float), _neurons, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, _SSBOs[0]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // Weights
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, _nWeights * sizeof(float), _weights, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, _SSBOs[1]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // Biases
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[2]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, _nBiases * sizeof(float), _biases, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, _SSBOs[2]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // Weight gradients
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[3]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, _nWeights * sizeof(float), _weightGradients, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, _SSBOs[3]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // Bias gradients
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[4]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, _nBiases * sizeof(float), _biasGradients, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, _SSBOs[4]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    // Forwarding Layers
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, _SSBOs[5]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, (nplLength - 1) * sizeof(forwardingLayer), _forwardingLayers, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, _SSBOs[5]);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


    // Base quad rendering init
    unsigned int _VAO, _VBO;
    glGenVertexArrays(1, &_VAO);
    glGenBuffers(1, &_VBO);
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(VERTS) * sizeof(float), VERTS, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);


    ImVec4 clearColor = {};
    bool isTraining = false;
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
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
        ImGui::SetNextWindowSize(io.DisplaySize, ImGuiCond_Always);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("FullScreenWindow", nullptr,
            ImGuiWindowFlags_NoTitleBar |
            ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar |
            ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::BeginTable("Basic AI Model", 2);
        ImGui::TableSetupColumn("Controls");
        ImGui::TableSetupColumn("Visuals");
        ImGui::TableHeadersRow();
        ImGui::TableNextColumn();
            if(ImGui::Button(isTraining? "Stop Training": "Train"))
                isTraining = !isTraining;
        ImGui::TableNextColumn();
            pos = ImGui::GetCursorScreenPos();
            ImVec2 size = ImGui::GetContentRegionAvail();
            int w, h;
            glfwGetFramebufferSize(_window, &w, &h);
            glViewport(pos.x, h - pos.y - size.y, size.x, size.y);

            // Compute
            if(isTraining) {
                glUseProgram(_computeModule);
                // Populate input layer with a random set from the dataset
                std::ifstream datafile(DATA_FILENAME);
                if (!datafile.is_open()) {
                    std::cerr << "Failed to open data file: " << DATA_FILENAME << std::endl;
                    break;
                }

                datafile.seekg(0, std::ios::end);
                int fileSize = datafile.tellg();
                
                int startByte = rand()%(fileSize); // Where to start looking for a line
                datafile.seekg(startByte, std::ios::beg);
                std::string line;
                std::getline(datafile, line);
                std::getline(datafile, line);
                
                if(!line.empty()){
                    std::stringstream ss(line);

                    std::string value;
                    for(int i=0; i < 4; i++){
                        getline(ss, value, ',');
                        #ifdef _DEBUG
                            std::cout << "Data file value: " << value << " of line: " << line << std::endl;
                        #endif
                        _neurons[i] = std::stof(value);
                    }
                    getline(ss, value, ',');
                    if(_targets.find(value) == _targets.end())
                        _targets[value] = _nTargets++;
                    int target = _targets[value];
                    
                    // glUniform1i(glGetUniformLocation(_computeModule, "targetIdx"), target);
                    for (int i = 0; i < nplLength - 1; ++i) {
                        glUniform1i(glGetUniformLocation(_computeModule, "layerIdx"), i);
                        glDispatchCompute((NODES_PER_LAYER[i+1] + 31)/32, 1, 1);
                        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);
                    }
                }
            }
        ImGui::EndTable();
        ImGui::End();
        ImGui::PopStyleVar();


        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            // Show visual representation of NN
            // glPolygonMode(GL_FRONT, GL_FILL);
            glUseProgram(_renderModule);
            glUniform1f(glGetUniformLocation(_renderModule, "aspectRatio"), size.x/size.y);
            glBindVertexArray(_VAO);
            // glActiveTexture(GL_TEXTURE0);
            // glBindTexture(GL_TEXTURE_BUFFER, _neuronTexBuffer);
            glDrawArrays(GL_TRIANGLES, 0, 6);
            // glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        /* Swap front and back buffers */
        glfwSwapBuffers(_window);

        glfwPollEvents();
    }

    // Clean
    delete[] _neurons;
    delete[] _weights;
    delete[] _biases;
    delete[] _forwardingLayers;
    _neurons = _weights = _biases = nullptr;
    _forwardingLayers = nullptr;

    // Clean glfw
    ImGui_ImplGlfw_Shutdown();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();

    glDeleteProgram(_renderModule);
    glDeleteBuffers(nBuffers, _SSBOs);
    glDeleteBuffers(1, &_VBO);


    return 0;
}
