#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <random>
#include <algorithm>
#include <sstream>

// Hyperparameters
const int embeddingDim = 100;
const int windowSize = 5;
const int numNegativeSamples = 5;
const float learningRate = 0.01;
const int numIterations = 100;

// Vocabulary and word embeddings
std::unordered_map<std::string, std::vector<float>> wordEmbeddings;
std::unordered_map<std::string, int> wordIndexMap;
std::vector<std::string> vocabulary;

// Data structures for training
std::vector<std::string> trainingData;

// Utility function to normalize a vector
void normalizeVector(std::vector<float>& vec) {
    float norm = 0.0;
    for (float val : vec) {
        norm += val * val;
    }
    norm = std::sqrt(norm);
    for (float& val : vec) {
        val /= norm;
    }
}

// Skip-gram training
void trainWordEmbeddings() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-0.5 / embeddingDim, 0.5 / embeddingDim);

    // Initialize word embeddings randomly
    for (const std::string& word : vocabulary) {
        std::vector<float> embedding;
        for (int i = 0; i < embeddingDim; ++i) {
            embedding.push_back(dis(gen));
        }
        wordEmbeddings[word] = embedding;
    }

    // Training loop
    for (int iter = 0; iter < numIterations; ++iter) {
        for (const std::string& targetWord : trainingData) {
            int targetIndex = wordIndexMap[targetWord];
            std::vector<float>& targetEmbedding = wordEmbeddings[targetWord];

            for (int contextPos = -windowSize; contextPos <= windowSize; ++contextPos) {
                if (contextPos == 0 || targetIndex + contextPos < 0 || targetIndex + contextPos >= vocabulary.size()) {
                    continue;
                }

                const std::string& contextWord = vocabulary[targetIndex + contextPos];
                std::vector<float>& contextEmbedding = wordEmbeddings[contextWord];

                // Positive example
                float dotProduct = 0.0;
                for (int i = 0; i < embeddingDim; ++i) {
                    dotProduct += targetEmbedding[i] * contextEmbedding[i];
                }
                float prediction = 1.0 / (1.0 + std::exp(-dotProduct));
                float error = prediction - 1.0;

                // Update target embedding
                for (int i = 0; i < embeddingDim; ++i) {
                    targetEmbedding[i] -= learningRate * error * contextEmbedding[i];
                }

                // Negative examples (randomly sampled)
                for (int k = 0; k < numNegativeSamples; ++k) {
                    int negativeIndex = gen() % vocabulary.size();
                    const std::string& negativeWord = vocabulary[negativeIndex];
                    std::vector<float>& negativeEmbedding = wordEmbeddings[negativeWord];

                    dotProduct = 0.0;
                    for (int i = 0; i < embeddingDim; ++i) {
                        dotProduct += targetEmbedding[i] * negativeEmbedding[i];
                    }
                    prediction = 1.0 / (1.0 + std::exp(-dotProduct));
                    error = prediction;

                    // Update target embedding
                    for (int i = 0; i < embeddingDim; ++i) {
                        targetEmbedding[i] -= learningRate * error * negativeEmbedding[i];
                    }
                }
            }
        }
    }
}

int main() {
    // Read training data
    std::stringstream file;
    file << "Hinkwaswo ndzi swi endlaka ndzi swi endla hikokwalaho ka ku swa boha leswaku ndzi swi endla. Leswi nga tano ndzi ta tsakela ku va ndzi nga voniwi nandzu loko ni endla hi ndlela leyi";
    std::string word;
    while (file >> word) {
        trainingData.push_back(word);
    }

    // Build vocabulary
    std::sort(trainingData.begin(), trainingData.end());
    trainingData.erase(std::unique(trainingData.begin(), trainingData.end()), trainingData.end());
    for (const std::string& word : trainingData) {
        wordIndexMap[word] = vocabulary.size();
        vocabulary.push_back(word);
    }

    // Train word embeddings
    trainWordEmbeddings();

    // Output word embeddings
    for (const auto& pair : wordEmbeddings) {
        std::cout << pair.first << ": ";
        for (float val : pair.second) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
