#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include "DenseLayer.h"

// Generates a random vector of specified size with values in the range [-1, 1]
std::vector<double> Layer::generateRandomVector(int size) {
    std::vector<double> res(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1, 1);
    for (double &val : res) {
        val = dist(gen);
    }
    return res;
}

// Initializes the weights of the layer with random values
void Layer::initWeights() {
    weights = generateRandomVector(layerSize * lastLayerSize);
}

// Initializes the biases of the layer with random values
void Layer::initBiases() {
    biases = generateRandomVector(layerSize);
}

// Sets the input values for the layer
void Layer::setInputs(const std::vector<double> &inputs) {
    this->input = inputs;
}

// Constructor to initialize the layer with its size and activation function
Layer::Layer(int layerSize, std::string activation)
        : layerSize(layerSize), activation(std::move(activation)), lastLayerSize(0){}

// Returns the size of the layer
int Layer::getLayerSize() const {
    return layerSize;
}

// Computes the outputs of the layer
void Layer::computeOutputs(bool isInputLayer) {
    if (isInputLayer) {
        outputs = input;
    } else {
        outputs.resize(layerSize);
        for (int i = 0; i < layerSize; ++i) {
            double sum = 0;
            for (int j = 0; j < lastLayerSize; ++j) {
                sum += weights[j + i * lastLayerSize] * input[j];
            }
            outputs[i] = sum + biases[i];
        }
    }
}

// Calculates the derivative of the activation function for the layer
std::vector<double> Layer::getLayerDerivative() {
    std::vector<double> res(layerSize);
    for (int i = 0; i < layerSize; ++i) {
        if (activation == "input") {
            res[i] = 1;
        } else if (activation == "relu") {
            res[i] = outputs[i] > 0 ? 1 : 0;
        } else if (activation == "sigmoid" || activation == "softmax") {
            res[i] = postactivation[i] * (1 - postactivation[i]);
        } else if (activation == "tanh") {
            res[i] = 1 - postactivation[i] * postactivation[i];
        } else if (activation == "linear") {
            res[i] = 1;
        } else {
            throw std::invalid_argument("Invalid activation function");
        }
    }
    return res;
}

// Calculates the derivative of the loss function with respect to the outputs
std::vector<double> Layer::getLossDerivative(const double &expectedOutput, const std::string &lossFunction) {
    std::vector<double> res(layerSize);
    if (lossFunction == "bin_crossentropy") {
        for (int i = 0; i < layerSize; ++i) {
            double epsilon = 1e-10;
            double output = postactivation[i];
            res[i] = (output == 0) ? (1 - expectedOutput) - expectedOutput / epsilon :
                     (output == 1) ? (1 - expectedOutput) / (1 - epsilon) - expectedOutput :
                     (1 - expectedOutput) / (1 - output) - expectedOutput / output;
        }
    } else if (lossFunction == "mse") {
        for (int i = 0; i < layerSize; ++i) {
            res[i] = postactivation[i] - expectedOutput;
        }
    } else if (lossFunction == "cat_crossentropy") {
        for (int i = 0; i < layerSize; ++i) {
            res[i] = (expectedOutput == i) ? postactivation[i] - 1 : postactivation[i];
        }
    } else {
        throw std::invalid_argument("Invalid loss function");
    }
    return res;
}

// Applies the activation function to the computed outputs
void Layer::computeActivation() {
    postactivation.resize(layerSize);
    for (int i = 0; i < layerSize; ++i) {
        if (activation == "input") {
            postactivation[i] = outputs[i];
        } else if (activation == "relu") {
            postactivation[i] = std::max(0.0, outputs[i]);
        } else if (activation == "sigmoid") {
            postactivation[i] = 1 / (1 + std::exp(-outputs[i]));
        } else if (activation == "tanh") {
            postactivation[i] = std::tanh(outputs[i]);
        } else if (activation == "linear") {
            postactivation[i] = outputs[i];
        }
    }
    if (activation == "softmax") {
        double sum = 0;
        for (double val : outputs) {
            sum += std::exp(val);
        }
        for (int i = 0; i < layerSize; ++i) {
            postactivation[i] = std::exp(outputs[i]) / sum;
        }
    }
}

// Updates the weights and biases of the layer based on the computed deltas and learning rate
void Layer::updateLayerWeightsAndBiases(const std::vector<double> &deltas, double learningRate) {
    for (int i = 0; i < layerSize; ++i) {
        for (int j = 0; j < lastLayerSize; ++j) {
            weights[j + i * lastLayerSize] -= learningRate * deltas[i] * input[j];
        }
        biases[i] -= learningRate * deltas[i];
    }
}

// Returns the outputs of the layer
std::vector<double> Layer::getOutputs() const {
    return outputs;
}

// Returns the weights of the layer
std::vector<double> Layer::getWeights() const {
    return weights;
}

// Returns the post-activation outputs of the layer
std::vector<double> Layer::getPostactivation() const {
    return postactivation;
}

// Sets the biases of the layer
void Layer::setBiases(const std::vector<double> &b) {
    biases = b;
}

// Sets the weights of the layer
void Layer::setWeights(const std::vector<double> &w) {
    weights = w;
}

// Sets the outputs of the layer
void Layer::setOutputs(const std::vector<double> &o) {
    outputs = o;
}

// Calculates the deltas for the layer based on the deltas from the next layer
std::vector<double> Layer::getDeltas(const std::vector<double> &lastdeltas, Layer &nextLayer) {
    std::vector<double> res(layerSize);
    std::vector<double> layerDerivative = getLayerDerivative();
    for (int i = 0; i < layerSize; ++i) {
        double sum = 0;
        for (int j = 0; j < nextLayer.getLayerSize(); ++j) {
            sum += nextLayer.getWeights()[j] * lastdeltas[j];
        }
        res[i] = sum * layerDerivative[i];
    }
    return res;
}

// Computes the loss of the layer
double Layer::getLoss(double expectedOutput, const std::string &lossFunction) {
    if (lossFunction == "bin_crossentropy") {
        double predictValue = postactivation[0];
        return -expectedOutput * std::log(predictValue) - (1 - expectedOutput) * std::log(1 - predictValue);
    } else if (lossFunction == "mse") {
        return std::pow(postactivation[0] - expectedOutput, 2);
    } else if (lossFunction == "cat_crossentropy") {
        double sum = 0;
        for (int i = 0; i < layerSize; ++i) {
            if (expectedOutput == i) {
                sum += std::log(postactivation[i]);
            }
        }
        return -sum;
    }
    return 0;
}

// Sets the size of the last layer
void Layer::setLastLayerSize(int size) {
    lastLayerSize = size;
}

// Initializes the layer by initializing its weights and biases
void Layer::initLayer() {
    initWeights();
    initBiases();
}

// Saves the layer configuration to a file
void Layer::saveLayer(std::ofstream &file) {
    if (!file.is_open()) {
        std::cerr << "Error: File is not open." << std::endl;
        return;
    }
    file << "#" << std::endl;
    file << layerSize << " " << activation << " " << lastLayerSize << std::endl;

    for (double bias : biases) {
        file << bias << " ";
    }
    file << std::endl;

    for (int i = 0; i < layerSize; ++i) {
        for (int j = 0; j < lastLayerSize; ++j) {
            file << weights[j + i * lastLayerSize] << " ";
        }
    }
    file << std::endl;
}

// Default constructor for Layer
Layer::Layer()
        : layerSize(0), activation("input"), lastLayerSize(0) {}

// Prints a summary of the layer configuration and returns the number of parameters
int Layer::layerSummary(int index) const {
    if (activation != "input") {
        std::cout << "Layer " << index << " | Size: " << layerSize << " | Activation: " << activation
                  << " | Params: " << layerSize * (lastLayerSize + 1) << std::endl;
    } else {
        std::cout << "Layer " << index << " | Size: " << layerSize << " | Activation: " << activation << std::endl;
    }
    std::cout << "------------------------------------------------" << std::endl;
    return layerSize * (lastLayerSize + 1);
}