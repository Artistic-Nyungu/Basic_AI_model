#include <iostream>
#include <cmath>

// Hyperparameters
const int NODES_PER_LAYER[] = {784, 6, 4, 6, 10};
const float LEARNING_RATE = 0.01;
const int MAX_ITERATIONS = 500;

/// Utility function to get the magnitude of an array
/// arr - array to be normalized
/// length - the length of the array
/// returns - a float representing the magnitude
float magnitude(const float* arr, int length){
    float squareSum = 0.0;
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

int main() {
    int _nNodes, _nWeights, _nBiases = 0;
    int nplLength = sizeof(NODES_PER_LAYER)/sizeof(int);
    for(int i=0; i < nplLength; i++){
        _nNodes += NODES_PER_LAYER[i];  // Count all the nodes
        if(i < nplLength - 1){
            _nWeights += NODES_PER_LAYER[i] * NODES_PER_LAYER[i+1]; // Count all the weights
            _nBiases += NODES_PER_LAYER[i+1]; // Count all the biases
        }
    }

    // All layer features will be mapped to a 1D array
    // Features of one layer will be sequential until the nth of the layer,
    // then the next will belong to the following layer
    float* _nodes = new float[_nNodes]; 
    float* _weights = new float[_nWeights];
    float* _biases = new float[_nBiases];

    std::cout << "Hello World!" << std::endl;

    // Clean
    delete[] _nodes;
    delete[] _weights;
    delete[] _biases;
    _nodes = _weights = _biases = nullptr;

    return 0;
}
