#version 460 core

struct Range {
    int begin;
    int end;
};

struct ForwardingLayer {
    Range neurons;
    Range weights;
    Range biases;
    int srcNeurons;   // Previous layer size (input for layer 0)
    int dstNeurons;   // Current layer size
};

layout(std430, binding = 0) buffer NeuronsBuffer { float neurons[]; };
layout(std430, binding = 1) buffer WeightsBuffer { float weights[]; };
layout(std430, binding = 3) buffer ForwardingLayersBuffer { ForwardingLayer layers[]; };

uniform int layersCount;

uniform vec4 backgroundCol = vec4(1.0);
uniform float neuronRadius = 0.02;
uniform float weightMinThickness = 0.0005;
uniform float weightMaxThickness = 0.007;
uniform float aspectRatio;  // width/height

uniform float minValNeurons;
uniform float minValWeights;
uniform float maxValNeurons;
uniform float maxValWeights;

in vec2 uv;
out vec4 FragColor;

void main() {
    FragColor = backgroundCol;
    
    // Adjust for aspect ratio
    vec2 aspectUV = vec2(uv.x * aspectRatio, uv.y);  // For max uv.x = 1.f so max aspectUV.x = 1 * aspectRatio

    // Add padding of radius on quad
    vec2 padding = vec2(neuronRadius);
    vec2 uvStart = padding;
    vec2 uvEnd = vec2(aspectRatio, 1.f) - padding;
    float isPadding = clamp(step(aspectUV.x, uvStart.x) + step(uvEnd.x, aspectUV.x) + step(aspectUV.y, uvStart.y) + step(uvEnd.y, aspectUV.y), 0.f, 1.f);
    FragColor = mix(FragColor, vec4(1.f, 0.f, 0.f, 1.f), isPadding);

    // Get Layer Index
    float layersStart = uvStart.x + neuronRadius * 2.f; // Reserve space for input layer
    float layersEnd = uvEnd.x;
    float layerWidth = (layersEnd - layersStart)/float(layersCount);
    int layerIdx = int((aspectUV.x - layersStart) / layerWidth);
    
    // if !isPadding (layerIdx < 0 || layerIdx > layersCount)
    //layerIdx < 0 || layerIdx > layersCount ? FragColor = vec4(1.f, 0.f, 1.f, 1.f) : FragColor = mix(FragColor, vec4(0.f, 0.f, 1.f, 1.f), float(layerIdx + 1)/float(layersCount + 1));
    float isValidLayer = clamp(step(layerIdx, 0) + step(layersCount - 1, layerIdx), 0.f, 1.f);
    float isInputLayer = step(aspectUV.x, layersStart);
    // Choose between blue and magenta, then choose opacity, the determine whether to display or not (!isPadding && !isInputLayer)
    FragColor = mix(FragColor, mix(FragColor, mix(vec4(1.f, 0.f, 1.f, 1.f), vec4(0.f, 0.f, 1.f, 1.f), isValidLayer), float(layerIdx + 1)/float(layersCount + 1)), (1.f - isPadding) * (1.f - isInputLayer));

    // TODO: Get Neuron Index on current layer
}