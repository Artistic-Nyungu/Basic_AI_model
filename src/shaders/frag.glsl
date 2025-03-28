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

uniform vec4 backgroundCol = vec4(1.0);
uniform float neuronRadius = 0.02;
uniform float weightMinThickness = 0.0005;
uniform float weightMaxThickness = 0.001;
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
    vec2 aspectUV = uv;
    aspectUV.x *= aspectRatio;
    
    // Draw input layer (layer 0's source neurons)
    float inputSpacing = 1.0 / (layers[0].srcNeurons + 1);
    for(int i = 0; i < layers[0].srcNeurons; i++) {
        vec2 pos = vec2(0.1 * aspectRatio, inputSpacing * (i + 1));
        float dist = distance(aspectUV, pos);
        if(dist < neuronRadius + neuronRadius * 0.05) {
            float activation = neurons[i];
            float alpha = smoothstep(minValNeurons, maxValNeurons, abs(activation));
            FragColor = dist < neuronRadius? mix(FragColor, vec4(0.0, 0.0, 1.0, alpha), alpha): vec4(0.f, 0.f, 0.f, 1.f);
            return;
        }
    }

    // Draw hidden + output layers and weights
    float layerSpacing = 0.9 / (layers.length() + 1);
    for(int layerIdx = 0; layerIdx < layers.length(); layerIdx++) {
        ForwardingLayer layer = layers[layerIdx];
        
        // Current layer position
        float layerX = 0.1 + layerSpacing * (layerIdx + 1);
        layerX *= aspectRatio;
        
        // Draw current layer neurons
        float neuronSpacing = 1.0 / (layer.dstNeurons + 1);
        for(int n = 0; n < layer.dstNeurons; n++) {
            vec2 pos = vec2(layerX, neuronSpacing * (n + 1));
            float dist = distance(aspectUV, pos);
            if(dist < neuronRadius + neuronRadius * 0.05) {
                float activation = neurons[layer.neurons.begin + n];
                float alpha = smoothstep(minValNeurons, maxValNeurons, abs(activation));
                FragColor = dist < neuronRadius? mix(FragColor, vec4(0.0, 0.0, 1.0, alpha), alpha): vec4(0.f, 0.f, 0.f, 1.f);
                return;
            }
        }

        // Draw weights from previous layer
        float prevLayerX = layerIdx == 0 ? 0.1 * aspectRatio : (0.1 + layerSpacing * layerIdx) * aspectRatio;
        int prevNeurons = layer.srcNeurons;
        float prevSpacing = 1.0 / (prevNeurons + 1);
        
        for(int src = 0; src < prevNeurons; src++) {
            for(int dst = 0; dst < layer.dstNeurons; dst++) {
                int weightIdx = layer.weights.begin + dst * prevNeurons + src;
                float weight = weights[weightIdx];
                
                vec2 start = vec2(prevLayerX, prevSpacing * (src + 1));
                vec2 end = vec2(layerX, neuronSpacing * (dst + 1));
                
                // Line calculation
                vec2 dir = end - start;
                float length = length(dir);
                vec2 dirNorm = dir / length;
                
                // Weight visualization
                float thickness = (weightMaxThickness - weightMinThickness) * smoothstep(minValWeights, maxValWeights, weight) + weightMinThickness;
                vec2 closestPoint = start + clamp(dot(aspectUV - start, dirNorm), 0.0, length) * dirNorm;
                float distToLine = distance(aspectUV, closestPoint);
                
                if(distToLine < thickness) {
                    vec3 color = weight > 0.0 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                    float alpha = smoothstep(minValWeights, maxValWeights, abs(weight));
                    FragColor = mix(FragColor, vec4(color, alpha * 0.7), alpha);
                    return;
                }
            }
        }
    }
}